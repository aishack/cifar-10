/*
The MIT License (MIT)

Copyright (c) 2015 Utkarsh Sinha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <vector>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

using namespace std;

// Constants used throughout the project
#define NUM_TRAINING_IMAGES 50000
#define NUM_PATCHES         400000
#define NUM_CENTROIDS       1600
#define PATCH_SIZE          6

// Record the execution time of some code, in milliseconds.
#define DECLARE_TIMING(s)  int64 timeStart_##s; double timeDiff_##s; double timeTally_##s = 0; int countTally_##s = 0
#define START_TIMING(s)    timeStart_##s = cvGetTickCount()
#define STOP_TIMING(s)     timeDiff_##s = (double)(cvGetTickCount() - timeStart_##s); timeTally_##s += timeDiff_##s; countTally_##s++
#define GET_TIMING(s)      (double)(timeDiff_##s / (cvGetTickFrequency()*1000.0))
#define GET_AVERAGE_TIMING(s)   (double)(countTally_##s ? timeTally_##s/ ((double)countTally_##s * cvGetTickFrequency()*1000.0) : 0)
#define CLEAR_AVERAGE_TIMING(s) timeTally_##s = 0; countTally_##s = 0

struct training_t {
    std::vector<cv::Mat> images;
    std::vector<uchar> labels;
};

bool file_exists(const char* name) {
    struct stat buffer;
    return (stat(name, &buffer) == 0);
}

cv::Mat convert_vector_to_mat(std::vector<cv::Mat> patches) {
    const unsigned int num_patches = patches.size();
    const unsigned int patch_area = PATCH_SIZE * PATCH_SIZE * 3;
    cv::Mat ret = cv::Mat(num_patches, patch_area, CV_32FC1);

    for(int i=0;i<num_patches;i++) {
        cv::Mat tmp = patches[i].reshape(1, 1);
        cv::Mat tmp2;
        tmp.convertTo(tmp2, CV_32FC1);
        tmp.copyTo(ret(cv::Rect(0, i, patch_area, 1)));
    }

    return ret;
}

cv::Mat whiten_patches(cv::Mat patches) {
    printf("Normalizing patches ...\n");
    const unsigned int num_patches = patches.rows;
    const unsigned int patch_area = patches.cols;
    cv::Mat ret = cv::Mat(num_patches, patch_area, CV_32FC1, cv::Scalar(0));

    // Calculate the average patch
    cv::Mat mean = cv::Mat(1, patch_area, CV_32FC1);
    for(int i=0;i<num_patches;i++) {
        mean += patches(cv::Rect(0, i, patch_area, 1));
    }
    mean = mean / num_patches;

    // Calculate the variance of the patch
    cv::Mat var = cv::Mat(1, patch_area, CV_32FC1);
    for(int i=0;i<num_patches;i++) {
        cv::Mat tmp = patches(cv::Rect(0, i, patch_area, 1)) - mean;
        cv::Mat output;
        cv::pow(tmp, 2, output);

        var += output;
    }
    var /= num_patches;
    cv::Mat std_dev;
    cv::sqrt(var, std_dev);

    cv::Mat normalized = cv::Mat(num_patches, patch_area, CV_32FC1, cv::Scalar(0));
    for(int i=0;i<num_patches;i++) {
        cv::Mat out;
        cv::divide(patches.row(i) - mean, std_dev, out);
        out.copyTo(normalized.row(i));
    }

    // ZCA whitening
    printf("Whitening patches ...\n");
    cv::Mat covariance, mean2;
    cv::calcCovarMatrix(normalized, covariance, mean2, CV_COVAR_COLS, CV_32FC1);

    mean2 = cv::Mat(1, patch_area, CV_32FC1, cv::Scalar(0));
    for(int i=0;i<num_patches;i++) {
        mean2 += normalized.row(i);
    }
    mean2 /= num_patches;

    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covariance, eigenvalues, eigenvectors);

    cv::Mat eigenvectors_tr;
    cv::transpose(eigenvectors, eigenvectors_tr);

    cv::Mat t1 = cv::Mat::diag(eigenvalues + 0.1f);
    cv::Mat t2;
    cv::sqrt(t1, t2);
    cv::Mat t3 = eigenvectors_tr * t2 * eigenvectors;

    cv::Mat tmp = cv::Mat(num_patches, patch_area, CV_32FC1, cv::Scalar(0));
    for(int i=0;i<num_patches;i++) {
        cv::Mat t4 = normalized.row(i) - mean2;
        t4.copyTo(tmp.row(i));
    }


    ret = tmp * t3;

    return ret;
}

std::vector<cv::Mat> pick_random_patches(training_t *train) {
    printf("Picking %d random patches ...\n", NUM_PATCHES);
    std::vector<cv::Mat> ret;
    unsigned int count = 0;
    const unsigned int num_images = train->images.size();

    while(count < NUM_PATCHES) {
        // Pick a random image
        unsigned int pick = (int)(((double)rand() / RAND_MAX) * num_images);
        cv::Mat img = train->images[pick];

        // Pick a random coordinate to pick a patch from
        const unsigned int x = (32-PATCH_SIZE+1) * ((double)rand()/RAND_MAX);
        const unsigned int y = (32-PATCH_SIZE+1) * ((double)rand()/RAND_MAX);

        // Extract a PATCH_SIZE*PATCH_SIZE image patch
        cv::Mat patch = img(cv::Rect(x, y, PATCH_SIZE, PATCH_SIZE)).clone();
        ret.push_back(patch);
        count++;
    }

    return ret;
}

std::vector<cv::Mat> evaluate_centroids(cv::Mat patches) {
    printf("Evaluating common patches ...\n");

    DECLARE_TIMING(kmeans);
    START_TIMING(kmeans);
    // Now, we run kmeans clustering on these 1d "feature vectors"
    cv::Mat centroids, labels;
    cv::TermCriteria tc(10e-6, 100, cv::TermCriteria::COUNT+cv::TermCriteria::EPS);
    cv::kmeans(patches, NUM_CENTROIDS, labels, tc, 20, cv::KMEANS_PP_CENTERS, centroids);
    STOP_TIMING(kmeans);

    printf("Running kmeans too %d seconds\n", (int)GET_TIMING(kmeans));

    // We need to convert these 1d "patches" into the expected 6x6x3 size
    std::vector<cv::Mat> ret;
    for(int i=0;i<NUM_CENTROIDS;i++) {
        cv::Mat tmp = centroids(cv::Rect(0, i, PATCH_SIZE * PATCH_SIZE * 3, 1));
        ret.push_back(tmp.reshape(3, PATCH_SIZE).clone());
    }

    return ret;
}

training_t* load_data() {
    // The dataset contains exactly 50,000 patches
    std::vector<cv::Mat> images;
    std::vector<uchar> labels;

    // Temporary space to read each image
    uchar *tmp = new uchar[3072];
    cv::Mat channel_red, channel_green, channel_blue;
    std::vector<cv::Mat> channels;

    // Load the five training data sets
    printf("Loading dataset into memory\n");
    for(int i=1;i<=5;i++) {
        char filename[50];
        sprintf(filename, "./data_batch_%d.bin", i);
        FILE *fp = fopen(filename, "r");

        // Each file contains exactly 10k images
        for(int j=0;j<10000;j++) {
            // The first byte of each image is the label
            uchar label;
            fread(&label, 1, 1, fp);
            labels.push_back(label);

            // The next 32*32*3 bytes are the red, green and blue planes
            // respectively
            fread(tmp, 1, 3072, fp);

            channel_red   = cv::Mat(32, 32, CV_8UC1, tmp+0);
            channel_green = cv::Mat(32, 32, CV_8UC1, tmp+1024);
            channel_blue  = cv::Mat(32, 32, CV_8UC1, tmp+2048);

            channels.push_back(channel_red);
            channels.push_back(channel_green);
            channels.push_back(channel_blue);
            cv::Mat combined;

            cv::merge(channels, combined);
            images.push_back(combined);

            channel_red.release();
            channel_green.release();
            channel_blue.release();
            channels.clear();
        }

        fclose(fp);
        printf("   ... finished file #%d\n", i);
    }

    training_t *ret = new training_t();
    ret->images = images;
    ret->labels = labels;
    return ret;
}

// The main function starts here
int main(int argc, char* argv[]) {
    training_t *train = load_data();

    std::vector<cv::Mat> centroids;
    const char *file_centroids = "centroids.yaml";
    if(file_exists(file_centroids)) {
        cv::FileStorage fs(file_centroids, cv::FileStorage::READ);
        fs["centroids"] >> centroids;
        fs.release();
    } else {
        std::vector<cv::Mat> patches_vec = pick_random_patches(train);
        cv::Mat patches = convert_vector_to_mat(patches_vec);
        cv::Mat patches_whitened = whiten_patches(patches);
        std::vector<cv::Mat> centroids = evaluate_centroids(patches_whitened);

        cv::FileStorage fs(file_centroids, cv::FileStorage::WRITE);
        fs << "centroids" << centroids;
        fs.release();
    }

    return 0;
}

