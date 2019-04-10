#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

class Pixel
{
  public:
    uchar b, g, r; // 8 bit, 0-255

    Pixel(uchar b, uchar g, uchar r)
    {
        this->b = b;
        this->g = g;
        this->r = r;
    }
};

class KMeans
{
  private:
    std::vector<Pixel> clusterCentres;
    cv::Mat image;
    cv::Mat labels;
    int K;

  public:
    // Intialise cluster centres as random pixels from the image
    KMeans(cv::Mat image, int K)
    {
        this->image = image;
        this->K = K;
        labels = cv::Mat::zeros(cv::Size(image.rows, image.cols), CV_8UC1);
        for (int i = 0; i < K; i++)
        {
            int randRow = random(image.rows - 1);
            int randCol = random(image.cols - 1);
            cv::Vec3b bgr_pixel = image.at<cv::Vec3b>(randRow, randCol);
            uchar b = bgr_pixel[0];
            uchar g = bgr_pixel[1];
            uchar r = bgr_pixel[2];
            clusterCentres.push_back(Pixel(b, g, r));
        }
        assignNewClusterCentres();
    }
    void train(int iterations)
    {
        std::cout << "Training...\n";
        for (int i = 0; i < iterations; i++)
        {
            computeCentroids();
            assignNewClusterCentres();
            std::cout << "Train step " << i << "done" << std::endl;
        }
    }

    void convert()
    {
        for (int r = 0; r < image.rows; r++)
        {
            for (int c = 0; c < image.cols; c++)
            {

                Pixel p = clusterCentres.at(labels.at<uchar>(r, c));
                cv::Vec3b bgr_pixel(p.b, p.g, p.r);
                image.at<cv::Vec3b>(r, c) = bgr_pixel;
            }
        }
        std::cout << "showing..." << std::endl;
        cv::imshow("CompressedImage", image);
        cvWaitKey(0);
        cvDestroyAllWindows();
    }

  private:
    void assignNewClusterCentres()
    {
        std::cout << image.rows * image.cols << std::endl;
        for (int r = 0; r < image.rows; r++)
        {
            for (int c = 0; c < image.cols; c++)
            {
                int centroidLabel = 0;
                uchar b, g, r1;
                cv::Vec3b bgr_pixel = image.at<cv::Vec3b>(r, c);
                b = bgr_pixel[0];
                g = bgr_pixel[1];
                r1 = bgr_pixel[2];
                double distance, min_dist;
                min_dist = euclideanDistance(clusterCentres[0].b, clusterCentres[0].g, clusterCentres[0].r, b, g, r1);
                for (int i = 1; i < K; i++)
                {
                    distance = euclideanDistance(clusterCentres[i].b, clusterCentres[i].g, clusterCentres[i].r, b, g, r1);
                    if (distance < min_dist)
                    {
                        min_dist = distance;
                        centroidLabel = i;
                        labels.at<uchar>(r, c) = (uchar)centroidLabel;
                    }
                }
            }
        }
    }

  private:
    void computeCentroids()
    {
        for (int i = 0; i < K; i++)
        {
            double mean_b = 0.0, mean_g = 0.0, mean_r = 0.0;
            int n = 0;
            for (int r = 0; r < image.rows; r++)
            {
                for (int c = 0; c < image.cols; c++)
                {
                    if (labels.at<uchar>(r, c) == i)
                    {
                        cv::Vec3b bgr_pixel = image.at<cv::Vec3b>(r, c);
                        mean_b += bgr_pixel[0];
                        mean_g += bgr_pixel[1];
                        mean_r += bgr_pixel[2];
                        n++;
                    }
                }
            }
            mean_b /= n;
            mean_g /= n;
            mean_r /= n;
            clusterCentres.at(i) = Pixel(mean_b, mean_g, mean_r);
        }
    }

    static double euclideanDistance(int x1, int y1, int c1, int x2, int y2, int c2)
    {
        return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(c1, c2));
    }

    static int random(int lim)
    {
        std::default_random_engine dre(std::chrono::steady_clock::now().time_since_epoch().count()); // provide seed
        std::uniform_int_distribution<int> uid{0, lim};                                              // help dre to generate nos from 0 to lim (lim included);
        return uid(dre);                                                                             // pass dre as an argument to uid to generate the random no
    }
};

int main(int argc, char **argv)
{
    cv::Mat imageOriginal;
    imageOriginal = cv::imread("rain_princess.jpg");
    if (imageOriginal.empty())
        return -1;
    KMeans kmeans(imageOriginal, 6);
    kmeans.train(10);
    kmeans.convert();

    return 0;
}
