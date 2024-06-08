#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
//#include <opencv2/opencv.hpp>

using namespace std;
//using namespace cv;

random_device rd;
mt19937 gen(rd());


pair<double, double> findMinMax(const vector<vector<double>>& vec) {
    double min = numeric_limits<double>::max();
    double max = numeric_limits<double>::min();

    for (const auto& innerVec : vec) {
        for (double value : innerVec) {
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }
    }

    return make_pair(min, max);
}

extern "C" {

// Function to convert a 2D vector of doubles to an OpenCV Mat (image)
//Mat vectorToImage(const vector<vector<double>>& data, double min, double max) {
//    int rows = data.size();
//    int cols = (rows > 0) ? data[0].size() : 0;
//
//    Mat image(rows, cols, CV_8UC1); // 8-bit single channel image
//
//    for (int i = 0; i < rows; ++i) {
//        for (int j = 0; j < cols; ++j) {
//            double val = data[i][j];
//            // Clipping the value to [0, 255] range
//            val = (val-min)/(max-min)*255;
//            image.at<uchar>(i, j) = static_cast<uchar>(val);
//        }
//    }
//
//    return image;
//}




double* linspace(double start, double stop, int num) {
    auto* result = new double[num];

    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }

    return result;
}


double to_start(double center, double extent){
    return center - extent/2;
}


double to_spacing(int x_size, double extent){
    return extent/x_size;
}


double linear_poly(double x, double a, double b){
    return x*a + b;
};

double inverse_linear_poly(double x, double a, double b){
    return x/a - b/a;
};

double quad_poly(double x, double a, double b, double c){
    return x*x*a + x*b + c;
};

double inverse_quad_poly(double x, double a, double b, double c){
    if (b*b-4*a*c+4*a*x > 0){
        return (-b+sqrt(b*b-4*a*c+4*a*x))/(2*a);
    } else {
        return 0;
    }
}

double cub_poly(double x, double a, double b, double c, double d){
    return x*x*x*a + x*x*b + x*c + d;
};


double inverse_cub_poly(double x, double a, double b, double c, double d)
{
    b /= a;
    c /= a;
    d = (d - x) / a;
    double q = (3.0 * c - pow(b, 2)) / 9.0;
    double r = (9.0 * b * c - 27.0 * d - 2.0 * pow(b, 3)) / 54.0;
    double disc = pow(q, 3) + pow(r, 2);
    double root_b = b / 3.0;
    vector<double> roots(3, 0);

    if (disc > 0) // One root
    {
        double s = r + sqrt(disc);
        s = ((s < 0) ? -pow(-s, (1.0 / 3.0)) : pow(s, (1.0 / 3.0)));
        double t = r - sqrt(disc);
        t = ((t < 0) ? -pow(-t, (1.0 / 3.0)) : pow(t, (1.0 / 3.0)));
        roots[0] = roots[1] = roots[2] = -root_b + s + t;
    }
    else if (disc == 0) // All roots real and at least two are equal
    {
        double r13 = ((r < 0) ? -pow(-r, (1.0 / 3.0)) : pow(r, (1.0 / 3.0)));
        roots[0] = -root_b + 2.0 * r13;
        roots[1] = roots[2] = -root_b - r13;
    }
    else // Only option left is that all roots are real and unequal
    {
        q = -q;
        double dum1 = q * q * q;
        dum1 = acos(r / sqrt(dum1));
        double r13 = 2.0 * sqrt(q);
        roots[0] = -root_b + r13 * cos(dum1 / 3.0);
        roots[1] = -root_b + r13 * cos((dum1 + 2.0 * M_PI) / 3.0);
        roots[2] = -root_b + r13 * cos((dum1 + 4.0 * M_PI) / 3.0);
    }

    sort(roots.begin(), roots.end());
//    cout << roots[1] << endl;
    return roots[1];
}


tuple<int, int, int, int> findMaxIndex(const vector<vector<vector<vector<double>>>>& vec) {
    if (vec.empty() || vec[0].empty())
        return {-1, -1, -1, -1}; // Return {-1, -1} if the vector is empty or contains empty vectors

    int max_h = 0;
    int max_i = 0;
    int max_j = 0;
    int max_k = 0;
    double max_val = vec[0][0][0][0];

    for (int h = 0; h < vec.size(); ++h) {
        for (int i = 0; i < vec[h].size(); ++i) {
            for (int j = 0; j < vec[h][i].size(); ++j) {
                for (int k = 0; k < vec[h][i][j].size(); ++k) {
                    if (vec[h][i][j][k] > max_val) {
                        max_val = vec[h][i][j][k];
                        max_h = h;
                        max_i = i;
                        max_j = j;
                        max_k = k;
                    }
                }
            }
        }
    }

    return {max_h, max_i, max_j, max_k};
}



void readCSV(const string& filename, vector<double>& column1, vector<double>& column2) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string token;
        vector<string> tokens;
        while (getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() == 2) {
            try {
                double val1 = stod(tokens[0]);
                double val2 = stod(tokens[1]);
                column1.push_back(val1);
                column2.push_back(val2);
            } catch (const invalid_argument& e) {
                cerr << "Invalid argument: " << e.what() << endl;
            } catch (const out_of_range& e) {
                cerr << "Out of range: " << e.what() << endl;
            }
        } else {
            cerr << "Invalid line: " << line << endl;
        }
    }

    file.close();
}


double findMax(double* arr, int size) {
    if (size <= 0 || arr == nullptr) {
        // Handle invalid input
        return 0.0; // Return a default value or handle error appropriately
    }

    double maxVal = arr[0]; // Initialize maxVal with the first element of the array

    for (int i = 1; i < size; ++i) {
        if (arr[i] > maxVal) {
            maxVal = arr[i]; // Update maxVal if a larger value is found
        }
    }

    return maxVal;
}


pair<vector<double>, vector<double>> truncateVectors(const vector<double>& vec1, const vector<double>& vec2, double a, double b) {
    vector<double> truncatedVec1;
    vector<double> truncatedVec2;

    // Iterate through the first vector
    for (double value : vec1) {
        if (value > a && value < b) {
            truncatedVec1.push_back(value);
            truncatedVec2.push_back(value);
        }
    }

    return make_pair(truncatedVec1, truncatedVec2);
}

double interpolate_lines(double cubic_fac, double quadratic_fac, double spacing, double wl_start, vector<double> lines,
                         vector<double> compspec_x, vector<double> compspec_y){
    double j;
    double x0;
    double x1;
    double y0;
    double y1;
    double y;
    double sum = 0;
    vector<double> temp_lines(lines.size());
    for (int i = 0; i < lines.size(); ++i) {
        temp_lines[i] = inverse_cub_poly(lines[i], cubic_fac, quadratic_fac, spacing, wl_start);

        if (temp_lines[i] < compspec_x[0] ||
            temp_lines[i] > compspec_x[compspec_x.size() - 1]) { continue; }


        auto it = lower_bound(compspec_x.begin(), compspec_x.end(), temp_lines[i]);

        // 'it' points to the first element not less than temp_line
        j = distance(compspec_x.begin(), it);

        // Linear interpolation
        if (j == 0) {
            y = 0;//compspec_y[0];
        } else if (j == compspec_x.size() - 1) {
            y = 0;//compspec_y[compspec_size - 1];
        } else {
            x0 = compspec_x[j - 1];
            x1 = compspec_x[j];
            y0 = compspec_y[j - 1];
            y1 = compspec_y[j];
            y = y0 + (y1 - y0) * ((temp_lines[i] - x0) / (x1 - x0));
        }
        sum += y;
    }
    return sum;
}


double interpolate_lines_chisq(double cubic_fac, double quadratic_fac, double spacing, double wl_start, vector<double> lines,
                         vector<double> compspec_x, vector<double> compspec_y){
    double j;
    double x0;
    double x1;
    double y0;
    double y1;
    double y;
    double sum = 0;
    vector<double> temp_lines(lines.size());
    int n_sum = 0;
    for (int i = 0; i < lines.size(); ++i) {
        temp_lines[i] = inverse_cub_poly(lines[i], cubic_fac, quadratic_fac, spacing, wl_start);

        if (temp_lines[i] < compspec_x[0] ||
            temp_lines[i] > compspec_x[compspec_x.size() - 1]) { sum += 1.; continue; }


        auto it = lower_bound(compspec_x.begin(), compspec_x.end(), temp_lines[i]);

        // 'it' points to the first element not less than temp_line
        j = distance(compspec_x.begin(), it);

        // Linear interpolation
        if (j == 0) {
            y = 0;//compspec_y[0];
        } else if (j == compspec_x.size() - 1) {
            y = 0;//compspec_y[compspec_size - 1];
        } else {
            x0 = compspec_x[j - 1];
            x1 = compspec_x[j];
            y0 = compspec_y[j - 1];
            y1 = compspec_y[j];
            y = y0 + (y1 - y0) * ((temp_lines[i] - x0) / (x1 - x0));
            n_sum++;
        }
        sum += (y-1)*(y-1);
    }
    if (sum != 0) {
        return sum;
    }
    else {
        return 1000000.;
    }

}


double levyRejectionSampling(double mu, double c, normal_distribution<>& n_dist, uniform_real_distribution<>& u_dist) {
    while (true) {
        double u = u_dist(gen);
        double v = n_dist(gen);

        // Calculate candidate x
        double x_candidate = mu + c / (v * v);

        // Calculate the acceptance probability
        double p = sqrt(c / (2 * M_PI)) * exp(-c / (2 * (x_candidate - mu))) / pow(x_candidate - mu, 1.5);

        // Generate a uniform random number for acceptance decision
        double u2 = u_dist(gen);
        double u3 = u_dist(gen);

        // Accept or reject the candidate
        if (u2 <= p) {
            if (u3 > 0.5){
                return x_candidate;
            }
            else{
                return -x_candidate;
            }
        }
    }
}


void fitlines_mkcmk(const double* compspec_x, const double* compspec_y, const double* lines,
                                                      int lines_size, int compspec_size, int n_samples, double wl_start,
                                                      double spacing, double quadratic_fac, double cubic_fac,
                                                      double wl_stepsize, double spacing_stepsize, double quad_stepsize,
                                                      double cub_stepsize, double wl_cov, double spacing_cov,
                                                      double quad_cov, double cub_cov, const string& outname){
    vector<double> temp_lines(lines_size);
    vector<double> compspec_x_vec(compspec_x, compspec_x + compspec_size);
    vector<double> compspec_y_vec(compspec_y, compspec_y + compspec_size);
    vector<double> lines_vec(lines, lines + lines_size);
    double this_correlation = interpolate_lines_chisq(cubic_fac, quadratic_fac, spacing, wl_start,
                                                lines_vec, compspec_x_vec, compspec_y_vec);


    const double wl_lo = wl_start-wl_cov/2;
    const double wl_hi = wl_start+wl_cov/2;
    const double spacing_lo = spacing-spacing_cov/2;
    const double spacing_hi = spacing+spacing_cov/2;
    const double quad_lo = quadratic_fac-quad_cov/2;
    const double quad_hi = quadratic_fac+quad_cov/2;
    const double cub_lo = cubic_fac-cub_cov/2;
    const double cub_hi = cubic_fac+cub_cov/2;

    double step_st;
    double step_sp;
    double step_quad;
    double step_cub;
    double step_num;
    double next_correlation;

    normal_distribution<> step_dis(0, wl_stepsize);
    normal_distribution<> space_dis(0, spacing_stepsize);
    normal_distribution<> quad_dis(0, quad_stepsize);
    normal_distribution<> cub_dis(0, cub_stepsize);
    normal_distribution<> standard_normal(0, 1);
    uniform_real_distribution<> step_dist(0., 1.);
    ofstream stat_outfile(outname);
    int n_accepted = 0;

    for (int j = 0; j < n_samples; ++j) {
        if (j % 100000 == 0 && j != 0){
            cout << static_cast<double>(n_accepted)/static_cast<double>(j+1) << endl;
        }
        step_st = step_dis(gen);
        step_sp = space_dis(gen);
        step_quad = quad_dis(gen);
        step_cub = cub_dis(gen);
        step_num = step_dist(gen);
//        step_st =   levyRejectionSampling(0, wl_stepsize, standard_normal, step_dist);
//        step_sp =   levyRejectionSampling(0, spacing_stepsize, standard_normal, step_dist);
//        step_quad = levyRejectionSampling(0, quad_stepsize, standard_normal, step_dist);
//        step_cub =  levyRejectionSampling(0, cub_stepsize, standard_normal, step_dist);

        if(!(wl_lo < wl_start+step_st && wl_start+step_st < wl_hi && spacing_lo < spacing+step_sp && spacing+step_sp < spacing_hi &&
            quad_lo < quadratic_fac+step_quad && quadratic_fac+step_quad < quad_hi && cub_lo < cubic_fac+step_cub && cubic_fac+step_cub < cub_hi)){
            stat_outfile << setprecision(8) <<  wl_start << "," << spacing << "," << quadratic_fac << "," << cubic_fac << "," << this_correlation << "\n";
            continue;
        }

        next_correlation = interpolate_lines_chisq(cubic_fac+step_cub, quadratic_fac+step_quad, spacing+step_sp, wl_start+step_st,
                                                   lines_vec, compspec_x_vec, compspec_y_vec);

//        cout << "Xirel " << next_correlation/this_correlation << " Triggers: " << (step_num < (20*next_correlation/this_correlation)-19) << endl;
//        cout << "This correlation: " << this_correlation << " Next correlation: " << next_correlation << endl;

        if (next_correlation < this_correlation){
            wl_start += step_st;
            spacing += step_sp;
            quadratic_fac += step_quad;
            cubic_fac += step_cub;
            this_correlation = next_correlation;
            n_accepted++;
        }
        else if (step_num < (-100.*next_correlation/this_correlation)+101.){
            wl_start += step_st;
            spacing += step_sp;
            quadratic_fac += step_quad;
            cubic_fac += step_cub;
            this_correlation = next_correlation;
            n_accepted++;
        }

        stat_outfile << setprecision(8)  <<  wl_start << "," << spacing << "," << quadratic_fac << "," << cubic_fac << "," << this_correlation << "\n";

    }
    stat_outfile.close();
}


tuple <double, double, double, double> fitlines(double* compspec_x, const double* compspec_y,
                                                double* lines, int lines_size, int compspec_size,
                                                double center, double extent, double quadratic_ext,
                                                double cubic_ext, size_t c_size, size_t s_size,
                                                size_t q_size, size_t cub_size, double c_cov,
                                                double s_cov, double q_cov, double cub_cov, double zoom_fac, int n_refinement){
    cout << "Fitting lines..." << endl;
//    const size_t c_size = 100;   // 100;
//    const size_t s_size = 50;    // 50;
//    const size_t q_size = 100;   // 100;
//    const size_t cub_size = 100; // 100;

//    double c_cov = 100.;
//    double s_cov = 0.05;
//    double q_cov = 2.e-5;
//    double cub_cov = 2.5e-10;
//    cout << "fitlines reached" << endl;

    double final_c = 0;
    double final_s = 0;
    double final_q = 0;
    double final_cub = 0;

    double x0;
    double x1;
    double y0;
    double y1;
    double y;

//    cout << "Compspec_size: "<< compspec_size << endl;
//    cout << "Lines_size: "<< lines_size << endl;
//    cout << "Center: " << center<< endl;
//    cout << "Extent: " << extent<< endl;

    double dcub;
    double dq;
    double dc;
    double ds;
    vector<double> compspec_x_vec(compspec_x, compspec_x + compspec_size);

    double q_cov_frac = q_cov / q_size;
    double cub_cov_frac = cub_cov / cub_size;
    double c_cov_frac = c_cov / c_size;
    double s_cov_frac = s_cov / s_size;

    double q_cov_neghalf = -q_cov / 2;
    double cub_cov_neghalf = -cub_cov / 2;
    double c_cov_neghalf = -c_cov / 2;
    double s_cov_neghalf = -s_cov / 2;

    int j;
    vector<vector<vector<vector<double>>>> fit_vals(cub_size, vector<vector<vector<double>>>(q_size, vector<vector<double>>(c_size, vector<double>(s_size))));
    double sum = 0.;

    vector<double> temp_lines(lines_size);

//    cout << "loop reached" << endl;
    for (int n=0; n < n_refinement; n++) {
        cout << "Loop " << n+1 << " out of " << n_refinement << endl;
        #pragma omp parallel for collapse(4) schedule(dynamic) private(sum, y, y0, x0, y1, x1, dq, dcub, dc, ds, j) firstprivate(temp_lines)
        for (int cub_ind = 0; cub_ind < cub_size; ++cub_ind) {
            for (int q_ind = 0; q_ind < q_size; ++q_ind) {
                for (int c_ind = 0; c_ind < c_size; ++c_ind) {
                    for (int s_ind = 0; s_ind < s_size; ++s_ind) {
                        // cout << center+dc << " " << extent+extent*ds << endl;
                        // cout << to_start(center+dc, extent+extent*ds) << endl;
                        sum = 0.;
                        dq = linear_poly(double(q_ind), q_cov_frac, q_cov_neghalf);
                        dcub = linear_poly(double(cub_ind), cub_cov_frac, cub_cov_neghalf);
                        dc = linear_poly(double(c_ind), c_cov_frac, c_cov_neghalf);
                        ds = linear_poly(double(s_ind), s_cov_frac, s_cov_neghalf);
                        for (int i = 0; i < lines_size; ++i) {
                            //cout << extent+extent*ds << endl;
                            //cout << extent+extent*ds << endl;
                            //cout << lines[i] << ", " << cubic_ext+dcub << ", " << quadratic_ext + dq << ", " << to_spacing(compspec_size, extent + extent * ds)<< ", " << to_start(center + dc, extent + extent * ds) << endl;
                            temp_lines[i] = inverse_cub_poly(lines[i],
                                                             cubic_ext+dcub,
                                                             quadratic_ext + dq,
                                                             to_spacing(compspec_size, extent + extent * ds),
                                                             to_start(center + dc, extent + extent *
                                                                                             ds));// Get the x position from the lines array
                            if (temp_lines[i] < compspec_x[0] ||
                                temp_lines[i] > compspec_x[compspec_size - 1]) { continue; }


                            auto it = lower_bound(compspec_x_vec.begin(), compspec_x_vec.end(), temp_lines[i]);

                            // 'it' points to the first element not less than temp_line
                            j = distance(compspec_x_vec.begin(), it);

                            // Linear interpolation
                            if (j == 0) {
                                y = 0;//compspec_y[0];
                            } else if (j == compspec_size - 1) {
                                y = 0;//compspec_y[compspec_size - 1];
                            } else {
                                x0 = compspec_x[j - 1];
                                x1 = compspec_x[j];
                                y0 = compspec_y[j - 1];
                                y1 = compspec_y[j];
                                y = y0 + (y1 - y0) * ((temp_lines[i] - x0) / (x1 - x0));
                            }
//                            #pragma omp atomic
                            sum += y;
                        }
                        fit_vals[cub_ind][q_ind][c_ind][s_ind] = sum;
                    }
                }
            }
        }
        auto max_indices = findMaxIndex(fit_vals);
        int temp_ind = get<2>(max_indices);
        int temp_ind_2 = get<3>(max_indices);
        ofstream outFile("debug_"+to_string(n)+".txt");

        if (outFile.is_open()) {
            // Iterate over each row
            for (const auto& row : fit_vals) {
                // Iterate over each element in the row
                for (const auto& element : row) {
                    // Write the element to the file
                    outFile << element[temp_ind][temp_ind_2] << " ";
                }
                // Write newline character after each row
                outFile << "\n";
            }
            // Close the file
            outFile.close();
            cout << "Debug data has been written to debug_"+to_string(n)+".txt" << endl;
        } else {
            cerr << "Unable to open file!" << endl;
        }

        cout << "Indices:" << get<0>(max_indices) << "," << get<1>(max_indices) << "," << get<2>(max_indices) << "," << get<3>(max_indices) << endl;

        double d_final_c = linear_poly(double(get<2>(max_indices)), c_cov_frac, c_cov_neghalf);
        double d_final_s = linear_poly(double(get<3>(max_indices)), s_cov_frac, s_cov_neghalf);
        double d_final_q = linear_poly(double(get<1>(max_indices)), q_cov_frac, q_cov_neghalf);
        double d_final_cub = linear_poly(double(get<0>(max_indices)), cub_cov_frac, cub_cov_neghalf);

        cout << "Diffs:" << setprecision(numeric_limits<double>::digits10 + 1) << d_final_c << "," << d_final_s << "," << d_final_q << "," << d_final_cub << endl;

        final_c += d_final_c;
        final_s += d_final_s*(1+final_s);
        final_q += d_final_q;
        final_cub += d_final_cub;

        cout << "Final outputs:" << setprecision(numeric_limits<double>::digits10 + 1) << final_c << "," << final_s << "," << final_q << "," << final_cub << endl;

        center += d_final_c;
        extent *= (1 + d_final_s);
        quadratic_ext += d_final_q;
        cubic_ext += d_final_cub;

        cout << "New Centers:" << setprecision(numeric_limits<double>::digits10 + 1) << center << "," << extent << "," << quadratic_ext << "," << cubic_ext << endl;

        c_cov /= zoom_fac;
        s_cov /= zoom_fac;
        q_cov /= zoom_fac;
        cub_cov /= zoom_fac;

        q_cov_frac = q_cov / q_size;
        cub_cov_frac = cub_cov / cub_size;
        c_cov_frac = c_cov / c_size;
        s_cov_frac = s_cov / s_size;

        q_cov_neghalf = -q_cov / 2;
        cub_cov_neghalf = -cub_cov / 2;
        c_cov_neghalf = -c_cov / 2;
        s_cov_neghalf = -s_cov / 2;

    }

//    for (int i = 0; i < lines_size; ++i) {
//        cout << temp_lines[i] << ", ";
//    }
//    cout << endl;

    // Convert 2D vector to image
//    auto[min, max] = findMinMax(fit_vals[125]);
//    Mat image = vectorToImage(fit_vals[125], min, max);

    // Display the image
//    imshow("Image", image);
//    waitKey(0);
    return {final_cub, final_q, final_c, final_s};
}}


//int oldmain(){
//    auto start = chrono::high_resolution_clock::now();
//
//    double center = 4485.9;
//    double extent = 1700;
//    double quadratic_ext = -7e-6;
//    double cubic_ext = 1.5e-10;
//
//    cout << "Originals:" << center << "," << extent << "," << quadratic_ext << "," << cubic_ext << endl;
//
//    // The code you want to measure
//    static const size_t compspec_size = 2031;
//    static const size_t lines_size = 373;
//    double x[compspec_size];
//    double y[compspec_size];
//    double l[lines_size];
//
//
//    vector<double> temp1 = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0, 219.0, 220.0, 221.0, 222.0, 223.0, 224.0, 225.0, 226.0, 227.0, 228.0, 229.0, 230.0, 231.0, 232.0, 233.0, 234.0, 235.0, 236.0, 237.0, 238.0, 239.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 256.0, 257.0, 258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 264.0, 265.0, 266.0, 267.0, 268.0, 269.0, 270.0, 271.0, 272.0, 273.0, 274.0, 275.0, 276.0, 277.0, 278.0, 279.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0, 317.0, 318.0, 319.0, 320.0, 321.0, 322.0, 323.0, 324.0, 325.0, 326.0, 327.0, 328.0, 329.0, 330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 336.0, 337.0, 338.0, 339.0, 340.0, 341.0, 342.0, 343.0, 344.0, 345.0, 346.0, 347.0, 348.0, 349.0, 350.0, 351.0, 352.0, 353.0, 354.0, 355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 361.0, 362.0, 363.0, 364.0, 365.0, 366.0, 367.0, 368.0, 369.0, 370.0, 371.0, 372.0, 373.0, 374.0, 375.0, 376.0, 377.0, 378.0, 379.0, 380.0, 381.0, 382.0, 383.0, 384.0, 385.0, 386.0, 387.0, 388.0, 389.0, 390.0, 391.0, 392.0, 393.0, 394.0, 395.0, 396.0, 397.0, 398.0, 399.0, 400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 409.0, 410.0, 411.0, 412.0, 413.0, 414.0, 415.0, 416.0, 417.0, 418.0, 419.0, 420.0, 421.0, 422.0, 423.0, 424.0, 425.0, 426.0, 427.0, 428.0, 429.0, 430.0, 431.0, 432.0, 433.0, 434.0, 435.0, 436.0, 437.0, 438.0, 439.0, 440.0, 441.0, 442.0, 443.0, 444.0, 445.0, 446.0, 447.0, 448.0, 449.0, 450.0, 451.0, 452.0, 453.0, 454.0, 455.0, 456.0, 457.0, 458.0, 459.0, 460.0, 461.0, 462.0, 463.0, 464.0, 465.0, 466.0, 467.0, 468.0, 469.0, 470.0, 471.0, 472.0, 473.0, 474.0, 475.0, 476.0, 477.0, 478.0, 479.0, 480.0, 481.0, 482.0, 483.0, 484.0, 485.0, 486.0, 487.0, 488.0, 489.0, 490.0, 491.0, 492.0, 493.0, 494.0, 495.0, 496.0, 497.0, 498.0, 499.0, 500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0, 508.0, 509.0, 510.0, 511.0, 512.0, 513.0, 514.0, 515.0, 516.0, 517.0, 518.0, 519.0, 520.0, 521.0, 522.0, 523.0, 524.0, 525.0, 526.0, 527.0, 528.0, 529.0, 530.0, 531.0, 532.0, 533.0, 534.0, 535.0, 536.0, 537.0, 538.0, 539.0, 540.0, 541.0, 542.0, 543.0, 544.0, 545.0, 546.0, 547.0, 548.0, 549.0, 550.0, 551.0, 552.0, 553.0, 554.0, 555.0, 556.0, 557.0, 558.0, 559.0, 560.0, 561.0, 562.0, 563.0, 564.0, 565.0, 566.0, 567.0, 568.0, 569.0, 570.0, 571.0, 572.0, 573.0, 574.0, 575.0, 576.0, 577.0, 578.0, 579.0, 580.0, 581.0, 582.0, 583.0, 584.0, 585.0, 586.0, 587.0, 588.0, 589.0, 590.0, 591.0, 592.0, 593.0, 594.0, 595.0, 596.0, 597.0, 598.0, 599.0, 600.0, 601.0, 602.0, 603.0, 604.0, 605.0, 606.0, 607.0, 608.0, 609.0, 610.0, 611.0, 612.0, 613.0, 614.0, 615.0, 616.0, 617.0, 618.0, 619.0, 620.0, 621.0, 622.0, 623.0, 624.0, 625.0, 626.0, 627.0, 628.0, 629.0, 630.0, 631.0, 632.0, 633.0, 634.0, 635.0, 636.0, 637.0, 638.0, 639.0, 640.0, 641.0, 642.0, 643.0, 644.0, 645.0, 646.0, 647.0, 648.0, 649.0, 650.0, 651.0, 652.0, 653.0, 654.0, 655.0, 656.0, 657.0, 658.0, 659.0, 660.0, 661.0, 662.0, 663.0, 664.0, 665.0, 666.0, 667.0, 668.0, 669.0, 670.0, 671.0, 672.0, 673.0, 674.0, 675.0, 676.0, 677.0, 678.0, 679.0, 680.0, 681.0, 682.0, 683.0, 684.0, 685.0, 686.0, 687.0, 688.0, 689.0, 690.0, 691.0, 692.0, 693.0, 694.0, 695.0, 696.0, 697.0, 698.0, 699.0, 700.0, 701.0, 702.0, 703.0, 704.0, 705.0, 706.0, 707.0, 708.0, 709.0, 710.0, 711.0, 712.0, 713.0, 714.0, 715.0, 716.0, 717.0, 718.0, 719.0, 720.0, 721.0, 722.0, 723.0, 724.0, 725.0, 726.0, 727.0, 728.0, 729.0, 730.0, 731.0, 732.0, 733.0, 734.0, 735.0, 736.0, 737.0, 738.0, 739.0, 740.0, 741.0, 742.0, 743.0, 744.0, 745.0, 746.0, 747.0, 748.0, 749.0, 750.0, 751.0, 752.0, 753.0, 754.0, 755.0, 756.0, 757.0, 758.0, 759.0, 760.0, 761.0, 762.0, 763.0, 764.0, 765.0, 766.0, 767.0, 768.0, 769.0, 770.0, 771.0, 772.0, 773.0, 774.0, 775.0, 776.0, 777.0, 778.0, 779.0, 780.0, 781.0, 782.0, 783.0, 784.0, 785.0, 786.0, 787.0, 788.0, 789.0, 790.0, 791.0, 792.0, 793.0, 794.0, 795.0, 796.0, 797.0, 798.0, 799.0, 800.0, 801.0, 802.0, 803.0, 804.0, 805.0, 806.0, 807.0, 808.0, 809.0, 810.0, 811.0, 812.0, 813.0, 814.0, 815.0, 816.0, 817.0, 818.0, 819.0, 820.0, 821.0, 822.0, 823.0, 824.0, 825.0, 826.0, 827.0, 828.0, 829.0, 830.0, 831.0, 832.0, 833.0, 834.0, 835.0, 836.0, 837.0, 838.0, 839.0, 840.0, 841.0, 842.0, 843.0, 844.0, 845.0, 846.0, 847.0, 848.0, 849.0, 850.0, 851.0, 852.0, 853.0, 854.0, 855.0, 856.0, 857.0, 858.0, 859.0, 860.0, 861.0, 862.0, 863.0, 864.0, 865.0, 866.0, 867.0, 868.0, 869.0, 870.0, 871.0, 872.0, 873.0, 874.0, 875.0, 876.0, 877.0, 878.0, 879.0, 880.0, 881.0, 882.0, 883.0, 884.0, 885.0, 886.0, 887.0, 888.0, 889.0, 890.0, 891.0, 892.0, 893.0, 894.0, 895.0, 896.0, 897.0, 898.0, 899.0, 900.0, 901.0, 902.0, 903.0, 904.0, 905.0, 906.0, 907.0, 908.0, 909.0, 910.0, 911.0, 912.0, 913.0, 914.0, 915.0, 916.0, 917.0, 918.0, 919.0, 920.0, 921.0, 922.0, 923.0, 924.0, 925.0, 926.0, 927.0, 928.0, 929.0, 930.0, 931.0, 932.0, 933.0, 934.0, 935.0, 936.0, 937.0, 938.0, 939.0, 940.0, 941.0, 942.0, 943.0, 944.0, 945.0, 946.0, 947.0, 948.0, 949.0, 950.0, 951.0, 952.0, 953.0, 954.0, 955.0, 956.0, 957.0, 958.0, 959.0, 960.0, 961.0, 962.0, 963.0, 964.0, 965.0, 966.0, 967.0, 968.0, 969.0, 970.0, 971.0, 972.0, 973.0, 974.0, 975.0, 976.0, 977.0, 978.0, 979.0, 980.0, 981.0, 982.0, 983.0, 984.0, 985.0, 986.0, 987.0, 988.0, 989.0, 990.0, 991.0, 992.0, 993.0, 994.0, 995.0, 996.0, 997.0, 998.0, 999.0, 1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1011.0, 1012.0, 1013.0, 1014.0, 1015.0, 1016.0, 1017.0, 1018.0, 1019.0, 1020.0, 1021.0, 1022.0, 1023.0, 1024.0, 1025.0, 1026.0, 1027.0, 1028.0, 1029.0, 1030.0, 1031.0, 1032.0, 1033.0, 1034.0, 1035.0, 1036.0, 1037.0, 1038.0, 1039.0, 1040.0, 1041.0, 1042.0, 1043.0, 1044.0, 1045.0, 1046.0, 1047.0, 1048.0, 1049.0, 1050.0, 1051.0, 1052.0, 1053.0, 1054.0, 1055.0, 1056.0, 1057.0, 1058.0, 1059.0, 1060.0, 1061.0, 1062.0, 1063.0, 1064.0, 1065.0, 1066.0, 1067.0, 1068.0, 1069.0, 1070.0, 1071.0, 1072.0, 1073.0, 1074.0, 1075.0, 1076.0, 1077.0, 1078.0, 1079.0, 1080.0, 1081.0, 1082.0, 1083.0, 1084.0, 1085.0, 1086.0, 1087.0, 1088.0, 1089.0, 1090.0, 1091.0, 1092.0, 1093.0, 1094.0, 1095.0, 1096.0, 1097.0, 1098.0, 1099.0, 1100.0, 1101.0, 1102.0, 1103.0, 1104.0, 1105.0, 1106.0, 1107.0, 1108.0, 1109.0, 1110.0, 1111.0, 1112.0, 1113.0, 1114.0, 1115.0, 1116.0, 1117.0, 1118.0, 1119.0, 1120.0, 1121.0, 1122.0, 1123.0, 1124.0, 1125.0, 1126.0, 1127.0, 1128.0, 1129.0, 1130.0, 1131.0, 1132.0, 1133.0, 1134.0, 1135.0, 1136.0, 1137.0, 1138.0, 1139.0, 1140.0, 1141.0, 1142.0, 1143.0, 1144.0, 1145.0, 1146.0, 1147.0, 1148.0, 1149.0, 1150.0, 1151.0, 1152.0, 1153.0, 1154.0, 1155.0, 1156.0, 1157.0, 1158.0, 1159.0, 1160.0, 1161.0, 1162.0, 1163.0, 1164.0, 1165.0, 1166.0, 1167.0, 1168.0, 1169.0, 1170.0, 1171.0, 1172.0, 1173.0, 1174.0, 1175.0, 1176.0, 1177.0, 1178.0, 1179.0, 1180.0, 1181.0, 1182.0, 1183.0, 1184.0, 1185.0, 1186.0, 1187.0, 1188.0, 1189.0, 1190.0, 1191.0, 1192.0, 1193.0, 1194.0, 1195.0, 1196.0, 1197.0, 1198.0, 1199.0, 1200.0, 1201.0, 1202.0, 1203.0, 1204.0, 1205.0, 1206.0, 1207.0, 1208.0, 1209.0, 1210.0, 1211.0, 1212.0, 1213.0, 1214.0, 1215.0, 1216.0, 1217.0, 1218.0, 1219.0, 1220.0, 1221.0, 1222.0, 1223.0, 1224.0, 1225.0, 1226.0, 1227.0, 1228.0, 1229.0, 1230.0, 1231.0, 1232.0, 1233.0, 1234.0, 1235.0, 1236.0, 1237.0, 1238.0, 1239.0, 1240.0, 1241.0, 1242.0, 1243.0, 1244.0, 1245.0, 1246.0, 1247.0, 1248.0, 1249.0, 1250.0, 1251.0, 1252.0, 1253.0, 1254.0, 1255.0, 1256.0, 1257.0, 1258.0, 1259.0, 1260.0, 1261.0, 1262.0, 1263.0, 1264.0, 1265.0, 1266.0, 1267.0, 1268.0, 1269.0, 1270.0, 1271.0, 1272.0, 1273.0, 1274.0, 1275.0, 1276.0, 1277.0, 1278.0, 1279.0, 1280.0, 1281.0, 1282.0, 1283.0, 1284.0, 1285.0, 1286.0, 1287.0, 1288.0, 1289.0, 1290.0, 1291.0, 1292.0, 1293.0, 1294.0, 1295.0, 1296.0, 1297.0, 1298.0, 1299.0, 1300.0, 1301.0, 1302.0, 1303.0, 1304.0, 1305.0, 1306.0, 1307.0, 1308.0, 1309.0, 1310.0, 1311.0, 1312.0, 1313.0, 1314.0, 1315.0, 1316.0, 1317.0, 1318.0, 1319.0, 1320.0, 1321.0, 1322.0, 1323.0, 1324.0, 1325.0, 1326.0, 1327.0, 1328.0, 1329.0, 1330.0, 1331.0, 1332.0, 1333.0, 1334.0, 1335.0, 1336.0, 1337.0, 1338.0, 1339.0, 1340.0, 1341.0, 1342.0, 1343.0, 1344.0, 1345.0, 1346.0, 1347.0, 1348.0, 1349.0, 1350.0, 1351.0, 1352.0, 1353.0, 1354.0, 1355.0, 1356.0, 1357.0, 1358.0, 1359.0, 1360.0, 1361.0, 1362.0, 1363.0, 1364.0, 1365.0, 1366.0, 1367.0, 1368.0, 1369.0, 1370.0, 1371.0, 1372.0, 1373.0, 1374.0, 1375.0, 1376.0, 1377.0, 1378.0, 1379.0, 1380.0, 1381.0, 1382.0, 1383.0, 1384.0, 1385.0, 1386.0, 1387.0, 1388.0, 1389.0, 1390.0, 1391.0, 1392.0, 1393.0, 1394.0, 1395.0, 1396.0, 1397.0, 1398.0, 1399.0, 1400.0, 1401.0, 1402.0, 1403.0, 1404.0, 1405.0, 1406.0, 1407.0, 1408.0, 1409.0, 1410.0, 1411.0, 1412.0, 1413.0, 1414.0, 1415.0, 1416.0, 1417.0, 1418.0, 1419.0, 1420.0, 1421.0, 1422.0, 1423.0, 1424.0, 1425.0, 1426.0, 1427.0, 1428.0, 1429.0, 1430.0, 1431.0, 1432.0, 1433.0, 1434.0, 1435.0, 1436.0, 1437.0, 1438.0, 1439.0, 1440.0, 1441.0, 1442.0, 1443.0, 1444.0, 1445.0, 1446.0, 1447.0, 1448.0, 1449.0, 1450.0, 1451.0, 1452.0, 1453.0, 1454.0, 1455.0, 1456.0, 1457.0, 1458.0, 1459.0, 1460.0, 1461.0, 1462.0, 1463.0, 1464.0, 1465.0, 1466.0, 1467.0, 1468.0, 1469.0, 1470.0, 1471.0, 1472.0, 1473.0, 1474.0, 1475.0, 1476.0, 1477.0, 1478.0, 1479.0, 1480.0, 1481.0, 1482.0, 1483.0, 1484.0, 1485.0, 1486.0, 1487.0, 1488.0, 1489.0, 1490.0, 1491.0, 1492.0, 1493.0, 1494.0, 1495.0, 1496.0, 1497.0, 1498.0, 1499.0, 1500.0, 1501.0, 1502.0, 1503.0, 1504.0, 1505.0, 1506.0, 1507.0, 1508.0, 1509.0, 1510.0, 1511.0, 1512.0, 1513.0, 1514.0, 1515.0, 1516.0, 1517.0, 1518.0, 1519.0, 1520.0, 1521.0, 1522.0, 1523.0, 1524.0, 1525.0, 1526.0, 1527.0, 1528.0, 1529.0, 1530.0, 1531.0, 1532.0, 1533.0, 1534.0, 1535.0, 1536.0, 1537.0, 1538.0, 1539.0, 1540.0, 1541.0, 1542.0, 1543.0, 1544.0, 1545.0, 1546.0, 1547.0, 1548.0, 1549.0, 1550.0, 1551.0, 1552.0, 1553.0, 1554.0, 1555.0, 1556.0, 1557.0, 1558.0, 1559.0, 1560.0, 1561.0, 1562.0, 1563.0, 1564.0, 1565.0, 1566.0, 1567.0, 1568.0, 1569.0, 1570.0, 1571.0, 1572.0, 1573.0, 1574.0, 1575.0, 1576.0, 1577.0, 1578.0, 1579.0, 1580.0, 1581.0, 1582.0, 1583.0, 1584.0, 1585.0, 1586.0, 1587.0, 1588.0, 1589.0, 1590.0, 1591.0, 1592.0, 1593.0, 1594.0, 1595.0, 1596.0, 1597.0, 1598.0, 1599.0, 1600.0, 1601.0, 1602.0, 1603.0, 1604.0, 1605.0, 1606.0, 1607.0, 1608.0, 1609.0, 1610.0, 1611.0, 1612.0, 1613.0, 1614.0, 1615.0, 1616.0, 1617.0, 1618.0, 1619.0, 1620.0, 1621.0, 1622.0, 1623.0, 1624.0, 1625.0, 1626.0, 1627.0, 1628.0, 1629.0, 1630.0, 1631.0, 1632.0, 1633.0, 1634.0, 1635.0, 1636.0, 1637.0, 1638.0, 1639.0, 1640.0, 1641.0, 1642.0, 1643.0, 1644.0, 1645.0, 1646.0, 1647.0, 1648.0, 1649.0, 1650.0, 1651.0, 1652.0, 1653.0, 1654.0, 1655.0, 1656.0, 1657.0, 1658.0, 1659.0, 1660.0, 1661.0, 1662.0, 1663.0, 1664.0, 1665.0, 1666.0, 1667.0, 1668.0, 1669.0, 1670.0, 1671.0, 1672.0, 1673.0, 1674.0, 1675.0, 1676.0, 1677.0, 1678.0, 1679.0, 1680.0, 1681.0, 1682.0, 1683.0, 1684.0, 1685.0, 1686.0, 1687.0, 1688.0, 1689.0, 1690.0, 1691.0, 1692.0, 1693.0, 1694.0, 1695.0, 1696.0, 1697.0, 1698.0, 1699.0, 1700.0, 1701.0, 1702.0, 1703.0, 1704.0, 1705.0, 1706.0, 1707.0, 1708.0, 1709.0, 1710.0, 1711.0, 1712.0, 1713.0, 1714.0, 1715.0, 1716.0, 1717.0, 1718.0, 1719.0, 1720.0, 1721.0, 1722.0, 1723.0, 1724.0, 1725.0, 1726.0, 1727.0, 1728.0, 1729.0, 1730.0, 1731.0, 1732.0, 1733.0, 1734.0, 1735.0, 1736.0, 1737.0, 1738.0, 1739.0, 1740.0, 1741.0, 1742.0, 1743.0, 1744.0, 1745.0, 1746.0, 1747.0, 1748.0, 1749.0, 1750.0, 1751.0, 1752.0, 1753.0, 1754.0, 1755.0, 1756.0, 1757.0, 1758.0, 1759.0, 1760.0, 1761.0, 1762.0, 1763.0, 1764.0, 1765.0, 1766.0, 1767.0, 1768.0, 1769.0, 1770.0, 1771.0, 1772.0, 1773.0, 1774.0, 1775.0, 1776.0, 1777.0, 1778.0, 1779.0, 1780.0, 1781.0, 1782.0, 1783.0, 1784.0, 1785.0, 1786.0, 1787.0, 1788.0, 1789.0, 1790.0, 1791.0, 1792.0, 1793.0, 1794.0, 1795.0, 1796.0, 1797.0, 1798.0, 1799.0, 1800.0, 1801.0, 1802.0, 1803.0, 1804.0, 1805.0, 1806.0, 1807.0, 1808.0, 1809.0, 1810.0, 1811.0, 1812.0, 1813.0, 1814.0, 1815.0, 1816.0, 1817.0, 1818.0, 1819.0, 1820.0, 1821.0, 1822.0, 1823.0, 1824.0, 1825.0, 1826.0, 1827.0, 1828.0, 1829.0, 1830.0, 1831.0, 1832.0, 1833.0, 1834.0, 1835.0, 1836.0, 1837.0, 1838.0, 1839.0, 1840.0, 1841.0, 1842.0, 1843.0, 1844.0, 1845.0, 1846.0, 1847.0, 1848.0, 1849.0, 1850.0, 1851.0, 1852.0, 1853.0, 1854.0, 1855.0, 1856.0, 1857.0, 1858.0, 1859.0, 1860.0, 1861.0, 1862.0, 1863.0, 1864.0, 1865.0, 1866.0, 1867.0, 1868.0, 1869.0, 1870.0, 1871.0, 1872.0, 1873.0, 1874.0, 1875.0, 1876.0, 1877.0, 1878.0, 1879.0, 1880.0, 1881.0, 1882.0, 1883.0, 1884.0, 1885.0, 1886.0, 1887.0, 1888.0, 1889.0, 1890.0, 1891.0, 1892.0, 1893.0, 1894.0, 1895.0, 1896.0, 1897.0, 1898.0, 1899.0, 1900.0, 1901.0, 1902.0, 1903.0, 1904.0, 1905.0, 1906.0, 1907.0, 1908.0, 1909.0, 1910.0, 1911.0, 1912.0, 1913.0, 1914.0, 1915.0, 1916.0, 1917.0, 1918.0, 1919.0, 1920.0, 1921.0, 1922.0, 1923.0, 1924.0, 1925.0, 1926.0, 1927.0, 1928.0, 1929.0, 1930.0, 1931.0, 1932.0, 1933.0, 1934.0, 1935.0, 1936.0, 1937.0, 1938.0, 1939.0, 1940.0, 1941.0, 1942.0, 1943.0, 1944.0, 1945.0, 1946.0, 1947.0, 1948.0, 1949.0, 1950.0, 1951.0, 1952.0, 1953.0, 1954.0, 1955.0, 1956.0, 1957.0, 1958.0, 1959.0, 1960.0, 1961.0, 1962.0, 1963.0, 1964.0, 1965.0, 1966.0, 1967.0, 1968.0, 1969.0, 1970.0, 1971.0, 1972.0, 1973.0, 1974.0, 1975.0, 1976.0, 1977.0, 1978.0, 1979.0, 1980.0, 1981.0, 1982.0, 1983.0, 1984.0, 1985.0, 1986.0, 1987.0, 1988.0, 1989.0, 1990.0, 1991.0, 1992.0, 1993.0, 1994.0, 1995.0, 1996.0, 1997.0, 1998.0, 1999.0, 2000.0, 2001.0, 2002.0, 2003.0, 2004.0, 2005.0, 2006.0, 2007.0, 2008.0, 2009.0, 2010.0, 2011.0, 2012.0, 2013.0, 2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021.0, 2022.0, 2023.0, 2024.0, 2025.0, 2026.0, 2027.0, 2028.0, 2029.0, 2030.0};
//    for (int i = 0; i < temp1.size(); ++i) {
//        x[i] = temp1[i];
//    }
//
//// No max filter
////    vector<double> temp2 = {0.0, 2.431575886267183, 55.454621174124895, 146.00067841083614, 149.54371319820962, 120.8058663129168, 43.37933367763367, 15.630396758563847, 0.0, 16.22476178293823, 15.688422083702108, 8.900868189475887, 15.930908824405606, 4.548145330145189, 20.019424916267326, 17.345878057369873, 62.08127565529071, 146.17174815923727, 162.39331323475267, 133.50168222863, 103.51613589843942, 58.50952100580889, 42.812075081798184, 42.1802547243924, 14.581948495958386, 15.651902731279279, 6.0778521400109184, 3.971651974393353, 4.338743584187796, 0.10482909976963128, 0.0, 9.166621310448136, 25.44027591646818, 30.466829307013995, 46.24445056318996, 37.15058350676213, 19.309341446703456, 0.0, 6.438446188880562, 12.719155245638149, 24.43914688574887, 27.57057073334863, 29.995204819662604, 14.536087262720457, 24.96028418839819, 11.789470835238944, 0.0, 24.87737910564715, 47.72990725334762, 52.62652646395895, 57.72589732914025, 44.36649306662616, 13.045142974988039, 7.293043065345955, 14.938404623045699, 0.0, 5.551115749417704, 27.988483554337563, 29.24304006630132, 31.081948062353376, 31.945934285547082, 15.960583260934072, 0.0, 7.846360158076095, 7.138720541602879, 1.9203521200829528, 17.879097653727513, 64.88841595873782, 72.26043852672501, 98.21207983207842, 90.45094367577803, 44.74507869125091, 12.527456040641482, 15.498201838423029, 0.0, 4.600042768353887, 21.02463731387752, 22.236894599680454, 11.309539844496157, 6.417102493235689, 0.0, 14.846486658377898, 17.051344739899605, 23.32632266946689, 12.700200564118404, 0.0, 0.5237987121595324, 7.8464866583778985, 2.7756308542593615, 1.5988689736789183, 5.280677889162007, 4.412152320982386, 1.017448386382057, 8.24761768469898, 14.795952319354683, 17.08754680502011, 6.761639098830756, 9.203707464570698, 12.083158960246692, 0.0, 5.403014210279252, 16.203707464570698, 10.886796040293348, 19.43916030134301, 16.85921935095439, 28.646582153637837, 23.70374542190939, 34.90355414944338, 34.47330355448571, 10.984683430054929, 15.904824335221292, 0.0, 8.442942394366355, 17.83110969148902, 17.251256573358887, 25.352865769689515, 3.003469679435966, 16.74977367718793, 9.602341061911602, 0.0, 8.736326912911863, 10.36043212053437, 9.254695703481957, 6.791009140442156, 7.186314844872186, 14.901470522585441, 13.13776848571706, 2.1105765667564356, 0.0, 6.927164065263696, 4.096827121508568, 16.80867324566225, 8.596740097646034, 17.031533026444095, 11.683208755720443, 39.31224594188848, 51.701593794904284, 43.98791847211214, 56.54719754430289, 30.90624143433729, 6.189440912871305, 22.83644326763897, 25.87582398463087, 14.997044917748099, 11.786405979253232, 0.0, 2.7988725659874945, 13.48020617452471, 16.833771789618822, 29.2305712841885, 65.97011141597068, 142.02047476264988, 354.93249755197735, 607.7196689417228, 626.2687239215288, 448.86660418997826, 186.20624780304388, 79.83785012561407, 47.272660843935, 11.13787322521307, 6.567111517793592, 0.0, 12.892909222468461, 46.130564263234874, 90.5739952997501, 130.31913441183383, 101.28359284642193, 44.89258540101196, 0.8743771628196555, 54.97452349771743, 290.1100010573291, 396.38811579456865, 476.329045295005, 528.5414132645701, 464.69255490655223, 424.678027610483, 257.8927991154187, 86.10437585398267, 37.22144780710323, 1.0031626852664886, 0.0, 19.610306057788193, 81.8931368249514, 152.2142150276834, 198.15529712611988, 188.00948275044902, 128.60955714289594, 150.82455322716692, 255.77506542506967, 233.88023535150182, 141.3433483373101, 43.18649261703399, 20.019878495699913, 32.981599243111305, 14.797506500656255, 19.360844707905926, 0.0, 11.96787275240672, 55.94929686898013, 73.35669375542716, 66.35830370714689, 35.22040720771247, 29.461063694388486, 0.0, 17.003231347074006, 46.576765640746544, 82.06764625648384, 154.56380354943576, 174.45741596279368, 130.32354104880937, 66.4997666735294, 0.0, 44.34636170006843, 63.72285103138256, 57.42243345231964, 59.73196683479159, 6.858772207728634, 0.0, 9.134982441970351, 4.060799696750109, 25.32268794621814, 24.796027918156824, 16.15852239792639, 9.73628956481275, 42.58457619568617, 71.64995483848702, 61.287819982354904, 64.86050118991534, 26.815710536139704, 0.0, 6.264313315555228, 22.887111979303427, 24.501865313868166, 21.33734759770823, 27.978360680749347, 7.227065980647012, 4.758037901006219, 11.452774349137371, 6.669273435320292, 0.0, 0.5617107865807611, 10.43023244060123, 36.23070371480139, 28.195093510589004, 22.234048067766935, 23.398693727665886, 44.28923011201891, 41.45782589337932, 29.819216940312117, 14.960134377181703, 0.0, 25.380245070718274, 71.07956802612989, 58.328315250752894, 48.81585484229254, 39.605795424936105, 0.0, 21.701917417471577, 34.08795754745097, 25.747125189506505, 41.26464817563101, 30.208128184783163, 0.0, 3.080999862749195, 29.17952532096092, 24.507846025659546, 105.37788784132954, 136.29910435823854, 161.05309645060993, 57.172356499478155, 2.881482722924602, 106.41070452134477, 269.79148905898774, 311.7555081853675, 308.69857107373855, 125.17575429635463, 0.0, 40.008683875459155, 140.2845851784764, 212.33870160424794, 221.1650766883995, 165.39928086423652, 77.90033627130151, 42.87031773310832, 5.925354854023681, 14.814329124609685, 0.0, 1.8073943805350154, 87.30422192451488, 189.45905420632812, 189.70598476380542, 168.58165020609044, 63.75816749309138, 0.0, 2.6588978906793272, 7.711660438819536, 34.72201032587782, 57.189374884031395, 59.049417357372704, 28.65536649696719, 22.93134639261052, 9.64603605401362, 18.672827906609882, 24.340592342611444, 9.132282082438678, 0.0, 19.00073971839015, 99.38774066298151, 179.29183881179551, 185.48325010430926, 147.08524070106364, 34.61566826397825, 0.0, 10.30010479857333, 53.216155741692546, 79.27021951247525, 104.12838278605204, 101.69151967167136, 326.5041901807733, 566.8904846327507, 657.2387406631155, 529.9691022540653, 177.58507679540344, 48.26789813668893, 20.346329488804713, 161.92181568588012, 320.3298544238935, 0.0, 48.45617059970073, 78.86057580382067, 110.37700451027331, 96.66567663268142, 40.824582974280474, 5.36522729031185, 0.0, 15.33802187436595, 8.916559955468529, 4.113921975223775, 5.096320308372469, 6.792731116982168, 20.835661522672808, 71.68703802191044, 74.57222306823633, 76.42489070212696, 59.77680449410968, 49.511478578181595, 28.89607846752051, 42.95986994865598, 4.976247054590658, 71.24887222307598, 237.119316813443, 334.57960778852566, 311.1729804006113, 153.62521451448697, 22.607946627135107, 0.0, 7.3480596539782255, 2.8667714257203443, 21.786875312052643, 29.54115338942347, 34.408141788390594, 50.97518582846442, 27.75343562663238, 14.77087304370707, 3.6575301441132524, 0.0, 37.21169812194398, 51.864197434644666, 49.89483290996668, 26.941868559790237, 47.46765239922206, 45.27231396881507, 48.627481936584445, 20.03740180426439, 0.0, 6.822276506785784, 37.19403461301931, 70.19038837047697, 70.28653885857057, 51.683055619831975, 15.752938541175808, 0.977391821092624, 26.916915824574517, 30.192199206434225, 27.621957981064497, 10.892437351260469, 22.15349018636357, 13.858585949099052, 52.86503892957876, 151.97373826419062, 190.09745513944767, 214.46830872475334, 146.4009746051047, 52.80560803038452, 7.36693848106529, 0.0, 40.857718143132615, 47.990756270585734, 71.3057318949293, 101.20783033519683, 171.10612953802433, 143.6264487890719, 230.67809116554463, 162.5648448972263, 162.19377135798072, 136.21011715811073, 84.32103525856473, 64.89266761529143, 33.91506644869446, 20.24143547944732, 19.824385904197698, 22.80776718124639, 6.1888151155103515, 9.75360470826081, 0.0, 37.83075142585926, 45.552808725068644, 46.53932245229635, 64.64304151951137, 81.61343973109115, 66.56848591037601, 184.45894543963504, 271.18987312049285, 345.1489168018247, 560.3239804445689, 720.7189333123558, 671.0175349341512, 463.82125518206453, 139.80978465967496, 43.48684151981797, 44.93967288653448, 17.920988159119133, 0.0, 23.62339097584345, 58.81910321259443, 42.922639692988014, 44.13971252940678, 8.818973552582975, 0.0, 12.6336376320005, 20.279121064683295, 27.37835578714703, 53.701550033885724, 74.99305043460367, 37.94710693947809, 1.732992868785459, 5.390660583371073, 27.992519916345827, 70.5674570890742, 114.62448242033861, 123.23064082619635, 71.40592631358982, 34.55131780639567, 0.0, 10.464459907821038, 31.35092501615486, 42.62251599292949, 77.3368179858951, 48.77756858905741, 22.427140732080716, 60.12949036138889, 45.41342911844595, 66.68176449206953, 45.38960376669138, 0.0, 8.229265947338718, 36.75744556878999, 42.70445601467304, 35.93984023980943, 21.196663645062245, 5.48990998085128, 0.0, 12.873975698843878, 12.902900001585976, 13.79921416784282, 48.92925370504213, 49.08910099632271, 35.2904580270781, 72.80938505814106, 146.89603860757938, 151.58079601163445, 127.1191017936228, 90.24962717364201, 37.23637004111606, 19.745969868223256, 30.19087080942677, 11.09186132189825, 23.899155858653558, 11.652569545735332, 0.0, 9.425299838778528, 33.785490822848715, 39.927133023346414, 50.203339873398136, 41.831210118424906, 7.832091228642412, 0.5454762003148517, 0.0, 0.43384252781379473, 1.4252113762042882, 6.919332916052554, 51.4088929502534, 101.37453171077073, 115.74556584601646, 80.01267653897071, 21.625401169653287, 6.732963818392818, 0.0, 21.851222514058918, 17.046199816936223, 16.96164784866596, 18.81480008679864, 14.948089115871426, 9.385299168636493, 0.0, 6.539356367025448, 18.32371706740696, 32.66754592089546, 23.400236112597213, 18.650084033779876, 29.04761046646763, 119.55807589771416, 184.47634622025248, 216.34515909992433, 261.30078242505647, 243.31474844234071, 207.33417982454284, 224.4455550389266, 152.28426985877832, 73.26575059966444, 18.92646757493253, 11.787186090097293, 0.0, 19.786390235269437, 42.0004444673159, 137.50214324183185, 329.44348361736957, 760.2332936499656, 1247.3317211461633, 1444.8781698685257, 1368.6883112001792, 791.1282068364776, 443.5773799431181, 110.93784062617578, 38.90113504552846, 0.0, 4.542157547792385, 120.08931996850629, 370.5207859790464, 474.5240815147615, 483.8613679665798, 225.24491518203354, 91.49741461305985, 42.89198396217739, 8.92872118096102, 19.651669342277728, 20.582570986324072, 9.161882198290868, 0.0, 12.49091118306933, 114.00528792652585, 185.91915634055158, 188.37488079428022, 150.24594653822123, 45.076773329051775, 0.15958788894431564, 4.575062512836666, 0.0, 11.285394897507558, 53.926526887567434, 549.537891359194, 935.0728733102701, 1048.3914721446063, 795.7678787020945, 141.20480499298742, 0.0, 31.316244538582396, 41.31677967573478, 45.94872074165846, 61.119467161564444, 103.08990322888394, 237.41670378153822, 302.30604842496155, 273.37146986498306, 165.2301584315269, 95.15096648437611, 44.641833417051885, 16.67674499137638, 21.322687946218366, 0.0, 12.17276355685317, 8.627779865433695, 4.252488136006605, 7.707936511034404, 7.60823078301496, 6.415711175388651, 2.736222709975209, 0.0, 1.4217340684504052, 10.9486437971741, 29.83506540764438, 48.284044966142574, 60.198111292446356, 54.5143390048604, 0.8099824032037759, 11.006567857161826, 20.326707851904985, 24.809326954836934, 235.99575562577684, 357.5505134464945, 379.1657754270002, 285.3543056811177, 55.557632681982795, 13.691547076985898, 15.503659865874397, 0.0, 20.746230962835625, 2.1677417953469558, 1.4019344415405612, 8.930212919590758, 15.71036140139222, 10.424838300550391, 10.268832440055348, 6.6446926796572825, 0.42934704333515583, 2.2284498317733323, 14.261450912336613, 8.988847383681104, 0.0, 11.040938866334955, 26.319125668685956, 12.581438014573905, 14.376985677751463, 23.54478728153549, 20.99750180449678, 29.68105675113793, 40.46657577352721, 62.32444927492543, 67.24607918878019, 50.32312491856055, 85.33342956868228, 618.8474173467487, 1101.8221684355317, 1148.207235622817, 938.048557946043, 255.21813061156922, 104.29666996191713, 73.82297789913741, 37.563801188457774, 22.11244635337698, 13.963306626584654, 0.0, 9.22418780814587, 7.752217269915036, 14.125659810427805, 46.806825415404546, 130.95762125478814, 143.21558100242964, 120.15382662548745, 85.61600622951892, 36.451040386300974, 13.244720684331696, 13.147602584784181, 24.95741437800598, 56.27384412628567, 65.40358965557562, 95.04539511491248, 137.77788020433104, 176.09949076458884, 190.85579550602665, 260.5074982170945, 295.6013142058089, 620.7303127338971, 4682.809080452558, 7676.391618792795, 7823.84472890949, 5854.150863436284, 998.8208049717609, 20.375648072554895, 0.0, 888.1459101609387, 1055.3512280851014, 1061.7549056590765, 563.3205016992815, 150.88985472503532, 67.61147247718554, 42.96544655526441, 14.833015939148709, 0.0, 1.732512901651944, 8.45744517166986, 10.293926114162332, 0.8134312857084751, 25.65545271305791, 23.713844123484478, 28.178893404468, 34.91498078416839, 15.653223374616346, 15.401913036533642, 50.808653873441926, 187.27302609650633, 894.0707011470322, 1150.9461794663875, 1145.7415298235987, 662.4228402364456, 49.561621032001995, 0.0, 27.660896717038895, 38.61304416552207, 70.00780947394651, 143.91315102318458, 860.5723192364273, 2301.181533054077, 2611.658582032881, 2416.369115844911, 913.6141007766212, 29.7037539320454, 0.0, 18.769338153163517, 96.12284243224599, 1602.3081693941285, 3962.234363215894, 4463.5383161347345, 7304.538783435315, 7591.948767470602, 6901.367167123511, 5553.806708977409, 1260.7148415625716, 300.9045942245348, 120.1650530994666, 259.6593829137264, 385.2059616774043, 356.96402124212204, 293.2231717537711, 145.67929236235318, 81.50560420535021, 58.57079361665069, 43.349165129487346, 0.0, 5.16355908839364, 29.62034203176131, 270.9921612148819, 1200.5639746644308, 1420.5530277307914, 1445.4827107294548, 700.9771794216729, 176.96656338099524, 80.66846905529451, 0.0, 32.237722835014665, 50.21448152748417, 57.089917606257586, 34.013749313679455, 0.0, 20.309816371695433, 122.54124820676407, 298.9084773537543, 387.37164291197246, 316.45522402903975, 226.09192649223974, 47.52018474408965, 0.0, 2.1348408947271764, 8.109398886465215, 24.349966192205784, 47.780216787163454, 87.37414715403224, 294.8184947519933, 330.08239195648775, 324.3075872193947, 179.4274935705755, 25.626217240652295, 12.995139365451678, 29.598619312742585, 0.0, 13.799742948159519, 8.5275436083366, 16.882422276750503, 30.575135518585284, 45.76677840494517, 46.81169013251133, 52.268518762060694, 87.31514793714473, 242.78125025529243, 455.4196999946698, 485.06861929820775, 476.7736386194847, 833.8187844964561, 996.9251842935375, 867.2989381043997, 618.454044340923, 0.0, 134.39859117050923, 2125.121893014356, 3623.256617046123, 3827.1066943308188, 2943.314236311283, 596.1751497273774, 116.82596888136982, 12.38622364680009, 0.0, 750.859651617496, 1836.2745777819634, 1933.9185412656582, 1806.0831190200695, 470.63189570355985, 0.0, 30.966209331255868, 1272.35200845825, 2938.9903171129477, 3021.7268115478414, 2870.721670065093, 1255.155765022997, 0.0, 183.79629918631508, 2116.0789675935275, 2433.8761585619245, 2366.473256193438, 1253.2441106593396, 139.45060844752243, 106.49139307925543, 190.88349495003968, 166.45460804708728, 144.61155998053255, 77.97300132502005, 6.800339433420959, 25.58263251302037, 0.0, 29.02928691435386, 257.47380138150425, 543.2410052253272, 652.8745333677748, 736.3156707888106, 356.782180160144, 228.5094110508387, 54.747897669383065, 4.7629136568948525, 0.0, 5.001049328171348, 36.83531987310175, 63.020338931727565, 469.1283321601313, 1906.967925995022, 2199.0998433333025, 2108.387390328784, 958.6245249881445, 153.33824792388646, 65.02322629887703, 30.645868078229796, 0.0, 5.874457408272974, 86.69979503310333, 160.69230077449674, 358.0318156036594, 357.6691230529102, 253.73328917922072, 108.00359506993914, 27.88486442547378, 0.0, 3.2430029567701695, 37.164478082042024, 25.521917493274714, 27.44560923529957, 14.350582770961637, 7.133189059028837, 0.0, 1.44709222896563, 20.539595810380888, 24.88189266151926, 35.30216970637457, 32.86678239918433, 23.527843984957826, 104.02703378825254, 183.0851350932826, 212.67693960314, 200.67258096310775, 100.53781324369697, 0.0, 91.00448226922867, 470.117957861534, 663.6901446451379, 1149.3631446412019, 2302.3358753494304, 2217.3804566310528, 2333.7845938999326, 1201.8662579869956, 655.4833668147473, 330.635283904498, 215.53654887052198, 436.60522024384477, 302.16221560404574, 260.8810971759324, 107.46925308940695, 0.0, 43.52565291600308, 433.4241900793006, 853.5264312721918, 878.8156322675829, 811.3727491184079, 1233.2669195061949, 1224.8734788706934, 1183.65592285177, 623.9534378002654, 212.39856365027367, 470.9203204901228, 491.4687494867592, 436.3285686953077, 153.9190880150145, 46.7393836268418, 12.88212167741608, 11.638960565806656, 0.0, 24.53958187130138, 75.92169006811241, 91.31278846601958, 127.8224397678548, 106.68030680075503, 83.56800957055566, 90.88395182375757, 55.17790937846894, 0.0, 5.948951727079475, 47.01056809805573, 244.6431170798137, 247.0999985828671, 296.8625016362603, 662.8939550977493, 1044.3549765098749, 873.6178605911482, 646.4901770117863, 46.15524951392672, 0.0, 1503.1619075871054, 2699.8524988856466, 2728.584729990523, 2154.9141664741805, 328.2481414046965, 348.3272383486319, 344.5578642827595, 342.1465275080732, 0.0, 108.84206902647065, 483.03478936257216, 836.7042945510789, 832.9083641821885, 406.4215797699085, 251.84113405673656, 83.27590284785401, 48.109846866252155, 21.315516196481667, 54.36522856141937, 25.919471607996, 46.375179358592504, 38.201803669711126, 13.0, 11.241251895344021, 0.5948144494127519, 0.0, 16.684151005227932, 10.557296193130469, 1.9966624165463145, 158.3993051993118, 476.43168334957204, 602.34622137675, 611.3100298436625, 378.0967148377256, 39.7280428260658, 144.99588819398127, 149.02892556223946, 159.45127635612107, 72.38378407337268, 20.42385873072658, 0.6664406994095771, 1.8803063063260197, 1.1147406663401398, 6.360551289318664, 0.0, 2.4070352483463466, 12.650582258790791, 28.04203450842715, 50.07317477521315, 33.4088383604294, 38.906318079820494, 11.237980469328022, 0.0, 18.150941435473214, 40.91023630917903, 74.49003940596094, 77.61509662557546, 100.84224362493069, 111.3865723773938, 180.16218187306958, 899.8965287734409, 1516.2261043800613, 2462.773390406347, 2181.5004200784015, 1338.5616592832796, 689.2889371562574, 560.1079306963359, 546.5910230833756, 407.4744034949597, 181.20595563134566, 199.21029142273528, 201.12968896227767, 156.87849713725882, 55.05405285750976, 6.242075868386792, 0.0, 56.454926394105996, 132.8396480901822, 158.21808626848588, 117.23747661971652, 39.14106609020337, 24.183827799817664, 4.396178755757774, 9.005554404894838, 0.0, 8.744405381895376, 16.828176229236988, 69.39534283149851, 182.6098165912042, 204.39152356971135, 172.76468036971846, 66.48296384113019, 21.967457182893668, 35.26999104985725, 17.95010148518827, 0.0, 5.74116299654338, 14.754929440396154, 12.272789431406636, 36.812807552693585, 49.952007842694, 65.83870140612612, 182.03308088794597, 601.2150027617429, 616.6180204521049, 591.097269961751, 237.64252979478306, 0.4568626738337116, 26.957171151316288, 57.05466063367362, 49.35249636993831, 21.44917634451963, 4.1858272792085245, 0.0, 13.315945134748063, 5.06289296318073, 20.776758673947143, 27.849679610440035, 212.3491633463093, 448.3330486497771, 462.1292381475105, 405.7700928238612, 83.7246515531449, 4.139708534643887, 0.0, 16.473632161972546, 95.46449053241304, 776.9077096127548, 1007.364801332375, 1015.5972344355723, 593.1406129158167, 79.63256187272941, 36.00547675244934, 15.011394900654295, 0.0, 24.963175827115947, 52.836090685044155, 98.36819789568813, 105.49027870792588, 112.71016209946424, 50.86688427889976, 14.868428958102413, 1.3447489316424708, 0.0, 0.3512129730950164, 35.89189067695338, 45.086232493082434, 59.249695848961665, 66.04696758123941, 51.33092727674148, 24.001588386541016, 0.0, 40.3328812533191, 58.959423349620465, 53.09918736407462, 44.571573052853864, 21.69693323060619, 1.9882044883120216, 7.3737283433886205, 26.559746918686415, 61.476769873673675, 771.8380920253521, 1876.2609187227408, 1909.1542096880187, 1748.5152879626705, 384.46016094645324, 98.40603305592026, 55.86050118991534, 48.01635838359971, 28.43025059495767, 15.001452841986747, 5.56974940504233, 19.99210187538779, 24.89178392894769, 0.0, 183.3943512402168, 313.8192692040709, 325.7301670802101, 290.321345378208, 26.784152527380456, 23.170215719396992, 53.7816101013766, 33.77614470763547, 38.71824271706237, 0.0, 51.71824271706237, 108.96829976067556, 114.62397703902184, 86.20097658497366, 19.819275877071732, 0.0, 42.94119853686516, 41.55620602197246, 62.65012488614798, 63.59910276672872, 0.0, 38.38577256936287, 52.31861431411971, 74.24152227729292, 91.02393657412631, 106.1516420719854, 94.86559654435041, 751.366404158447, 2215.0561272312357, 2314.2428208606225, 2186.672599465855, 686.0304945505718, 134.93431288209945, 97.52161704750415, 57.44165815987026, 17.31391905800433, 29.07888838553663, 21.666969888817903, 19.322916409072832, 30.41713555611159, 42.21660372609176, 18.440880900908496, 25.349420770248116, 4.251282139313162, 2.57246961763758, 0.0, 2.807141036239045, 22.699578387499514, 11.275028332098373, 30.335975976323198, 36.844859534869784, 75.79929057041545, 68.87589389541517, 57.83170635881902, 7.588906523011929, 6.361062958090315, 0.0, 11.655753311623812, 10.813339048175294, 25.53372978941934, 29.602761097035, 24.37422393959332, 23.789422048859706, 21.89073271432494, 2.7750930322024487, 8.363406053007566, 22.565168419914016, 38.97817175711589, 278.2315126357803, 1041.3361245779477, 1122.1924589143982, 1096.2376781948167, 400.64496520451667, 29.969030852466858, 21.208139244103677, 0.0, 10.803642625205384, 15.30665362788477, 42.08040497977572, 34.074989680540284, 72.39340775727214, 717.301702405108, 1377.99119517188, 1408.4097246106835, 1226.1489383471758, 196.6040889283297, 36.389993592629935, 0.0, 62.79007585817817, 368.4343403503037, 382.881449861555, 430.289834676405, 345.8866057079108, 182.0906987006149, 235.41545855894014, 179.67806744374047, 60.03689434477769, 52.640793204245256, 10.510065865225897, 6.731289140242552, 12.165477406406808, 0.0, 31.595442978627716, 52.776192507510586, 246.57672813301497, 1900.0375770623123, 2359.4253498459725, 2300.516141103386, 1355.9546619469877, 112.47628164445837, 78.47148700921275, 34.9951510301878, 46.38834791266322, 83.98035420371548, 51.451719327937326, 39.32691316158616, 17.885766875207537, 0.0, 10.312254539946935, 3.804577327350671, 32.23606475438214, 17.422321556009365, 22.590688861945864, 13.711898557173981, 0.0, 13.280566454411655, 44.98722260487443, 100.14372401301648, 231.87511288455175, 225.39114210054208, 217.58774473394078, 64.58365697174168, 19.581112610140735, 13.083286391321735, 3.084751046817928, 0.0, 13.171834513761041, 17.936134028714378, 111.15019055125322, 219.95270238229205, 228.80419278948784, 209.64400134758625, 81.7347699782797, 0.0, 102.31928598254672, 88.51290064817249, 105.10416068506765, 105.34984424959293, 86.40440634999413, 81.8296180685129, 91.6500653765238, 123.40957980820622, 95.63323740299575, 47.66781817372839, 0.0, 15.609977001684456, 69.5381380980532, 70.08879543385888, 53.97413883704576, 28.72161898997979, 0.0, 24.344226705435858, 123.78424245679298, 1370.620717291966, 1666.9175374672288, 1711.866882472171, 1052.0392976741732, 93.3357472350815, 48.448407921468515, 16.202514510649735, 30.361267716804377, 25.097127370027465, 43.05950510894809, 41.92230028722861, 26.963581065038397, 29.93660099456497, 0.0, 35.377550016105715, 33.139044939930955, 32.82286155207544, 19.35346416926177, 0.0, 17.006639352866387, 15.001965715742244, 5.646199815196951, 14.806427378319086, 5.764144969615472, 25.024508353238843, 13.35212395575627, 20.552371150362205, 15.329896269178334, 22.478350715417264, 33.65796309044299, 29.933671690937445, 28.677068090657258, 33.357551091581854, 0.0, 9.133079956686288, 4.937057774875029, 34.43617519368445, 13.241113518735801, 22.232718006329378, 14.87923157135151, 26.44898097639384, 9.815342251881248, 17.916503666766403, 4.931838077362272, 9.819944237716527, 14.979383520547117, 9.476075442182491, 11.731237274006162, 49.53841897443999, 64.18005576228347, 55.74686055534562, 53.32028466830138, 53.53700160014341, 193.22600663758453, 440.8300188646499, 426.2503374526857, 392.38226690839815, 89.40148010598296, 0.0, 12.228930281494968, 12.458235871339639, 29.370843359297396, 20.00873175307288, 41.486793609293045, 41.79427364696653, 20.166132437666874, 14.372902999685039, 0.0, 22.222281224574317, 22.745727972688655, 41.606811419178484, 27.57241006640629, 17.344310646700023, 20.40035501044258, 17.007182287892192, 11.343020901877253, 16.760429576063416, 57.01067004899869, 43.75420315043971, 45.49194434726064, 41.17522903659301, 55.36324702711863, 107.06625854712752, 1130.195702543548, 1315.0409423263202, 1309.851627616849, 778.0780644863271, 22.720595117516268, 0.0, 120.32411212240095, 213.40192224684824, 177.47848801592568, 151.6810819620132, 50.15175992643435, 553.8485948452603, 654.2148915156251, 642.5203899704416, 346.3818939268076, 74.71506481399774, 32.5667115559188, 24.92997592545271, 25.60434921096089, 6.412132443534347, 18.62068199173177, 3.722322750333433, 0.0, 14.869181865275777, 26.08407887522867, 42.52805271896091, 35.670284574239986, 37.094238512156835, 8.355391529424878, 2.991457691057576, 0.0, 62.66237196045677, 107.2552311945824, 93.09406166177541, 103.68718948083506, 58.93045806737791, 23.43332966713433, 22.309664387213616, 39.612487981442655, 26.963315598352892, 3.676278109870509, 13.54461520227619, 46.37286318262568, 83.75468056337468, 129.53198871372365, 708.6823804087462, 2930.980174716129, 3063.5818591035913, 2957.571492231427, 1094.4713208138965, 227.25923528351177, 231.9019405682768, 185.6639731145492, 154.99900814647754, 45.209617747631455, 18.78911548600695, 24.68607959853057, 18.663670756854344, 35.424520979688396, 33.610709699327344, 19.78113518865962, 40.90467135230074, 23.460407926637117, 14.042961773393017, 25.498039372347876, 0.0, 8.14852774362953, 6.04327952508379, 10.659494417557198, 20.265788817971497, 15.272339899493318, 13.820916993814762, 6.119455427834055, 34.86352988774888, 6.490936519802517, 16.92272342382921, 12.604368976053138, 8.15035369282623, 21.740883746707823, 7.145765997294802, 14.463488664285705, 25.53792702340479, 6.880328191809895, 7.352098163553137, 0.0, 1.7168649785271555, 17.506058363204602, 10.342987230897052, 17.11574290568251, 28.31163473435049, 5.435543737328544, 20.543247614199572, 28.903714536347024, 33.47803078821585, 47.54043892207142, 618.50907155137, 1158.0686754279666, 1172.9381120007197, 1016.2872448485195, 188.38836058585707, 39.573404801233664, 26.235560620214073, 42.502011720162955, 4.022017784094942, 17.35547093789819, 19.437072382382894, 0.0, 9.136147937458873, 20.25795740755302, 10.255807461046516, 36.55765163975843, 26.9642763960112, 14.313033458353857, 21.232507754954668, 0.0, 13.353138646180014, 29.016734924882257, 21.75142883749777, 14.949817177555815, 36.45617108641272, 15.455320624793785, 12.89757593516174, 0.0, 22.131022482505614, 14.762045093316829, 36.564072342520376, 22.181434070449313, 35.15807994772308, 16.87862149866214, 18.242964126466177, 34.27633826871829, 14.109753810694883, 27.352103793967217, 41.09751564980456, 61.426589224592135, 39.34341508163516, 17.173908815565028, 0.0, 14.465750284622573, 17.191378263028582, 20.149805303695075, 17.281760749579917, 23.60765609228588, 16.360779919577226, 17.090093201275977, 87.56135917861639, 455.5087642379001, 546.6578436734576, 505.0507684317897, 283.449043048113, 55.70137648017021, 24.944462245729483, 34.7360221929448, 0.0, 22.194845931873033, 12.055463201724024, 10.229178653610916, 31.645864474077598, 12.00913718963784, 12.743917933897592, 13.25415522326739, 11.363462163325948, 5.223917380712464, 3.997611475695294, 22.328660339670023, 14.559475431425653, 4.026265763365473, 7.918158788127812, 48.93834773556409, 59.72108248458471, 72.785041645099, 60.97173812140613, 37.654112276403794, 7.874627439146934, 20.426333151493054, 40.7071807678783, 40.47169022865887, 52.66592942484817, 21.748666525228373, 3.1622335486028987, 88.44131107204976, 194.96132476592402, 207.7101752852068, 234.00671802854913, 701.1070201327739, 1972.7423070572672, 2039.4034789579068, 2095.3359208936163, 590.1008121131615, 111.82076040733864, 45.18076386214466, 0.0, 8.326221859002771, 47.92664131842389, 102.36284108697441, 258.34376236818457, 503.0362078402318, 472.67010460499796, 375.1543855385121, 148.37340260245082, 33.07910546115863, 11.491548586823683, 34.31459027464962, 81.96351400251956, 76.8605325861538, 112.1429111099867, 45.98448390233716, 15.768214343921045, 8.185513380129805, 0.0, 4.2836074159563395, 20.738969502516966, 27.73133775575502, 16.312415117194405, 47.052686956116986, 115.33384575230752, 127.17075011427505, 105.25799571809284, 39.98347071379044, 3.7303305316397655, 0.1689064534393765, 16.688624933613255, 9.813124152035243, 0.024219565089424577, 5.674815895701386, 1.7195949123990886, 27.97848754515985, 20.07638464799288, 15.53063956450842, 0.0, 17.456909664759223, 6.527908340677868, 43.129100851757585, 48.880515378924656, 141.68405636832995, 137.09821710592792, 142.70568325951604, 119.60776431773343, 60.70295737530523, 36.153766990308895, 22.215125297478835, 31.685541594070173, 9.073695766771834, 5.9969769512767925, 0.0, 18.504746176056415, 16.663691354082857, 19.10756264873953, 49.546490611384115, 143.45468117369774, 187.38239508735023, 178.40910489068256, 115.60135974720879, 16.185517550492023, 14.360079123673358, 67.7336812709791, 57.97410279591645, 36.33969244690388, 26.032794044053844, 0.0, 5.28587727202148, 29.34990709251315, 31.554130256230792, 39.51190470326128, 53.05335485245405, 25.161003428385357, 3.0195952577907974, 4.220155767835195, 22.6669430037216, 15.476199864130649, 0.0, 5.914733719615015, 22.69204154458953, 16.03212305055513, 41.591524862352344, 30.959961340224254, 15.849326867580658, 44.37939235629119, 166.3077491013928, 356.1025944238063, 369.2444069464409, 329.8774215170097, 117.47675821883149, 25.31188950333717, 0.0, 9.246515490390266, 22.398275378240214, 148.85918362961638, 824.7282357404765, 919.9719016203094, 958.768550613365, 455.4142033250455, 56.22644990342428, 15.892806730785423, 0.0, 6.112941948824528, 17.7606506926802, 54.42791059126694, 75.21047814604185, 88.15017405589674, 43.41953490841752, 35.18965028171624, 10.276054009691052, 0.4828024040375567, 0.0, 19.137208806597755, 20.373179311690137, 8.577209364074406, 3.0168304738131155, 3.7899985500753246, 17.584155855518475, 22.343708645394145, 15.678796787390183, 17.54504634663772, 41.29242330920465, 7.434608613914406, 0.0, 17.191666180229504, 36.88784782931543, 32.91057890857269, 30.880124975461513, 43.37954000466425, 6.583123587946147, 21.935636983918585, 18.004621956043593, 22.087000381062353, 20.193243394362753, 0.0, 23.186618726793313, 20.130422200491466, 29.069433784101193, 44.933376283883945, 16.83646761904538, 5.619912816717942, 0.0, 25.92730022969522, 3.035589106304087, 37.48201310184186, 55.35581298982106, 63.85794995905417, 40.497900133024814, 102.21673756828295, 175.55278020370815, 208.58185414394848, 213.1107004654798, 62.06631621770839, 18.39555611629612, 0.0, 19.583877043719895, 2.5411986174297, 30.39555611629612, 372.14152982318, 491.70328334305805, 485.12608539748044, 352.0273265864671, 27.178465214738026, 6.333248835100676, 25.195997796318352, 7.3106034604363686, 9.517929665904148, 19.748525688800783, 5.700331237421096, 9.10806235177597, 5.717545739742036, 0.0, 10.686696118139935, 7.948888824310643, 8.136435173270684, 7.785206738118859, 21.72982621184724, 36.39542887501443, 24.681038605229105, 18.937688486239722, 18.69444862376963, 21.049975544138533, 16.300812500276834, 14.655376118297681, 13.43025059495767, 0.0, 5.559350124416369, 14.247726821131664, 31.78455042284145, 36.54586047802468, 16.824110298787218, 35.19641106844938, 7.4076348453836545, 0.0, 9.330735180545616, 1.2378393719527594, 54.37593036063549, 146.93846329178587, 162.88906497948733, 155.19043049912807, 53.21634652105831, 23.426348477916235, 0.0, 57.970475662659965, 63.34511064146204, 76.39268312557306, 76.63006440784534, 35.14499487766875, 1.7855187772981935, 0.0, 94.7171666411839, 160.2449195348886, 218.4782677558253, 402.470343548398, 261.1212806599565, 263.58082138919167, 75.75826802546658, 10.882186752956613, 9.484813353809159, 11.54087174815777, 14.982932864634222, 9.651237689100071, 0.0, 11.500794014989651, 20.983292684304615, 46.10802390290428, 69.76640532854799, 45.802162594473884, 40.05753508499856, 43.07664339637358, 4.531436035250863, 6.825544723908479, 0.0, 34.51108895526477, 36.929729341571374, 14.36456952883077, 32.07540345603729, 33.76821295347213, 44.118597781771996, 20.237792095977284, 15.018838422128965, 12.442431377858838, 0.0, 24.392037717359017, 45.12201639603836, 46.618527364483725, 40.952571077622, 84.05017622932951, 157.24649582176357, 145.69311647260565, 152.85218308225512, 34.78266735565671, 12.623119901235896, 12.801139941869224, 19.192190345857398, 0.0, 10.992057182431154, 17.92430588788193, 29.796906716059084, 18.69236843636395, 19.264846285543854, 14.049561940727472, 1.2706485735886872, 0.0, 3.302710759280444, 22.819139997015554, 10.325676197114717, 14.694544726619824, 13.575171962413606, 15.209067940083742, 31.372213587224905, 277.37921571787115, 848.622874771755, 920.6136257372696, 880.5179950728775, 299.8845306010412, 19.84846089627922, 18.343857596455337, 13.319691615238753, 5.773362099511587, 0.0, 67.67399940020687, 111.19268007152141, 100.63467375177288, 57.3871933168507, 22.290245708157045, 0.0, 26.87567135207837, 21.22542115402075, 23.98931390892608, 34.09055376338529, 81.21560991737965, 74.59775951544134, 73.38563384644499, 23.5517490252264, 16.354838384649156, 11.882338955882233, 0.0, 18.35514713018688, 4.115174228907335, 10.010235631928708, 39.57938594696111, 23.439441430305124, 39.315373427698205, 8.268604313439027, 0.0, 14.656373431530028, 34.92926650237246, 58.694132813059014, 94.31591910615816, 271.9482233411384, 282.38715921971743, 235.114266148875, 127.40689655022607, 89.27479945674509, 138.72758829415284, 148.65116410314204, 107.78667411596757, 11.798052170293431, 0.0, 2.59091161960464, 41.505493326752685, 223.40974244855784, 249.22649570658336, 242.40296591685387, 128.24405036497842, 35.78555203944825, 31.74081332770197, 25.434710737548585, 0.0, 31.242572906999385, 20.728111150222958, 48.84573593136611, 52.567225309300284, 92.71412880600701, 603.8601366621278, 717.3185447452361, 710.2885558745215, 476.3020189514732, 114.52004535053584, 521.0307660733824, 669.8189883212463, 732.370019824818, 690.6253380859334, 380.51778886621105, 262.6094787403831, 224.73010952765344, 66.66712896067929, 51.26423345887906, 48.777979878215774, 0.0, 13.02528802467441, 35.03820275905264, 40.41514283575566, 120.95831988150803, 130.61695180025367, 93.27808551412409, 34.87802443373448, 17.36697998324553, 0.0, 36.601368502297646, 32.684620727540505, 21.530513152663843, 21.544429142930994, 59.63958054212003, 47.694048235784976, 670.9773734446089, 921.5418942052743, 903.4409063173432, 663.1548241038488, 15.880342757420749, 0.0, 9.110374050151677, 12.641888265974103, 24.078737925942278, 51.8882819167859, 35.42920713785793, 44.73522216538663, 47.18527428303355, 11.219280731953404, 0.0, 16.622453472801453, 11.341792505077592, 31.3988964143648, 24.80913546929196, 50.3851409741153, 171.17982195512195, 595.7557526138032, 928.7184998090208, 1415.3141410633013, 1124.9866587534702, 1372.961005740741, 1489.2677539262072, 1366.8088759165912, 1322.0907910489482, 326.99294403927115, 98.04256978566809, 53.90765169017459, 0.0, 37.19237767050731, 52.53470568171997, 74.71957513384041, 45.51654876094699, 19.030468140982066, 0.0, 15.86243828641932, 8.3992744543948, 301.98822176157046, 467.969156228613, 458.99568695581434, 420.7311570568231, 70.43185866539488, 28.784330855006374, 0.0, 80.27202431109663, 182.80572669459184, 164.24785625925438, 141.57249549718904, 22.171530842961374, 0.0, 7.185995386782679, 14.252958247603146, 82.89704467316778, 100.765893817304, 93.56774850265765, 26.89367608961038, 32.58468315151913, 29.82414274980465, 23.82773277772776, 20.399283424869736, 0.0, 14.80485339850702, 48.07973753589181, 49.553182749388725, 36.96784712358476, 16.08066090918919, 0.0, 18.230284699706772, 52.982401385383355, 52.91178086036916, 72.99045893034918, 46.44714672075361, 33.023037574526825, 0.0, 3.9894438860196715, 1.4985830148566492, 174.2214024745167, 230.6809918187214, 309.08304729412407, 229.51690511299194, 142.1304820071166, 78.15103042282931, 12.820138691532293, 26.211829153114195, 4.007045804932659, 20.64429599669961, 7.825415828917357, 0.0, 39.87525744250411, 17.97862816947054, 66.60727272675149, 64.50809964337577, 92.216790901796, 66.47007860416579, 60.51485329434399, 0.0, 72.5285813201408, 242.58883225716477, 254.47615849142971, 272.7788097247892, 128.62410014330658, 30.01238728510816, 31.60813243194866, 0.0, 6.529386486521389, 14.926421449543568, 26.314008338386884, 16.900366989851364, 2.446047252234848, 13.3053173857204, 54.41373410678011, 64.73412810160471, 55.726824002402054, 55.499019946892986, 38.4077369768861};
//// Max filter
//    //    vector<double> temp2 = {0.0, 0.015, 0.3415, 0.8991, 0.9209, 0.7439, 0.2671, 0.0963, 0.0, 0.0999, 0.0966, 0.0548, 0.0981, 0.028, 0.1233, 0.1068, 0.3823, 0.9001, 1.0, 0.8221, 0.6374, 0.3603, 0.2636, 0.2597,0.0898, 0.0964, 0.0374, 0.0245, 0.0267, 0.0006, 0.0, 0.0564, 0.1567, 0.1876, 0.2848, 0.2288, 0.1189, 0.0, 0.0396, 0.0783, 0.1505, 0.1698, 0.1847, 0.0895, 0.187, 0.1139, 0.0, 0.2533,0.486, 0.5358, 0.5878, 0.4517, 0.1328, 0.0743, 0.1521, 0.0, 0.0565, 0.285, 0.2978, 0.3165, 0.3253, 0.1625, 0.0, 0.0799, 0.0727, 0.0196, 0.182, 0.6607, 0.7358, 1.0, 0.921, 0.4556, 0.1276, 0.1578, 0.0, 0.0468, 0.2141, 0.2264, 0.1152, 0.0653, 0.0, 0.1512, 0.1736, 0.2375, 0.1293, 0.0, 0.0053, 0.0799, 0.0283, 0.0163, 0.0538, 0.0449, 0.0104, 0.084, 0.1507, 0.1889, 0.1511, 0.2637, 0.3462, 0.0, 0.1548, 0.4642, 0.3119, 0.5569, 0.483, 0.8207, 0.6791, 1.0, 0.9877, 0.3147, 0.4557, 0.0, 0.1633, 0.3449, 0.3051, 0.4483, 0.0531, 0.2962, 0.1698, 0.0, 0.1545, 0.1832, 0.1637, 0.1201, 0.1271, 0.2635, 0.1991, 0.0149, 0.0, 0.0114, 0.0065, 0.0268, 0.0137, 0.0272, 0.0187, 0.0628, 0.0826, 0.0702, 0.0903, 0.0493, 0.0099, 0.0365, 0.0413, 0.0239,0.0188, 0.0, 0.0045, 0.0215, 0.0269, 0.0467, 0.1053, 0.2268, 0.5667, 0.9704, 1.0, 0.7167, 0.2973, 0.1275, 0.0755, 0.0178, 0.0105, 0.0, 0.0206, 0.0737, 0.1446, 0.2081, 0.1617, 0.0717, 0.0014, 0.0878, 0.4632, 0.6329, 0.7606, 0.844, 0.742, 0.6781, 0.4118, 0.1375, 0.0594, 0.0016, 0.0, 0.0371, 0.1549, 0.288, 0.3749, 0.3557, 0.2433, 0.2854, 0.4839, 0.4425, 0.2674, 0.0817, 0.0379, 0.0624, 0.028, 0.0366, 0.0, 0.0226, 0.1059, 0.1579, 0.1563, 0.1366, 0.1152, 0.0, 0.0665, 0.1821, 0.3209, 0.6043, 0.6821, 0.5095, 0.26, 0.0, 0.1734, 0.2491, 0.2455, 0.3424, 0.0393, 0.0, 0.0524, 0.0233, 0.1452, 0.1421, 0.0926, 0.0558, 0.2441, 0.4107, 0.3513, 0.3718, 0.1537, 0.0, 0.0359, 0.1312, 0.1404, 0.1223, 0.2147, 0.1009, 0.0664, 0.1598, 0.0931,0.0, 0.0078, 0.1456, 0.3438, 0.2069, 0.1381, 0.1453, 0.275, 0.2574, 0.1105, 0.048, 0.0, 0.0814, 0.228, 0.1871, 0.1566, 0.127, 0.0, 0.0696, 0.1093, 0.0826, 0.1324, 0.0969, 0.0, 0.0099, 0.0936, 0.0786, 0.338, 0.4372, 0.5166, 0.1834, 0.0092, 0.3413, 0.8654, 1.0, 0.9902, 0.4015, 0.0, 0.1283, 0.45, 0.6811, 0.7094, 0.5305, 0.2499, 0.1375, 0.019, 0.0475, 0.0, 0.0058, 0.28, 0.6077, 0.6085, 0.5407, 0.2045, 0.0, 0.0081, 0.0136, 0.0528, 0.087, 0.0898, 0.0436, 0.0349, 0.0147, 0.0284, 0.037, 0.0139, 0.0, 0.0289, 0.1512, 0.2728, 0.2822, 0.2238, 0.0527, 0.0, 0.0157, 0.081, 0.1206, 0.1584, 0.1547, 0.4968, 0.8625, 1.0, 0.8064, 0.2702, 0.0734, 0.031, 0.2464, 0.4874, 0.0, 0.0737, 0.12, 0.1679, 0.1471, 0.0621, 0.0082, 0.0, 0.0233, 0.0136,0.0063, 0.0078, 0.0103, 0.0317, 0.1091, 0.1135, 0.1163, 0.091, 0.0753, 0.0545, 0.1284, 0.0149, 0.213, 0.7087, 1.0, 0.93, 0.4592, 0.0676, 0.0, 0.022, 0.0086, 0.0651, 0.0883, 0.1028,0.1524, 0.083, 0.0441, 0.0109, 0.0, 0.1112, 0.155, 0.1491, 0.0805, 0.1419, 0.1353, 0.1453, 0.0599, 0.0, 0.0204, 0.1112, 0.2256, 0.3047, 0.224, 0.0683, 0.0042, 0.1167, 0.1309, 0.1197,0.0472, 0.096, 0.0601, 0.2292, 0.6588, 0.8241, 0.9297, 0.6347, 0.2289, 0.0319, 0.0, 0.1771, 0.208, 0.2629, 0.2932, 0.3054, 0.1993, 0.3201, 0.2256, 0.225, 0.189, 0.117, 0.09, 0.0471,0.0281, 0.0275, 0.0316, 0.0086, 0.0135, 0.0, 0.0525, 0.0632, 0.0646, 0.0897, 0.1132, 0.0924, 0.2559, 0.3763, 0.4789, 0.7775, 1.0, 0.931, 0.6436, 0.194, 0.0603, 0.0624, 0.0249, 0.0,0.0328, 0.0816, 0.0596, 0.0612, 0.0122, 0.0, 0.0175, 0.0281, 0.038, 0.0745, 0.1041, 0.0527, 0.0024, 0.0075, 0.0388, 0.0979, 0.159, 0.171, 0.1064, 0.0745, 0.0, 0.0849, 0.2134, 0.2812,0.5102, 0.3218, 0.148, 0.3967, 0.2996, 0.4399, 0.2994, 0.0, 0.0543, 0.2425, 0.2817, 0.2371, 0.1398, 0.0362, 0.0, 0.0849, 0.0851, 0.091, 0.3228, 0.3238, 0.2328, 0.4803, 0.9691, 1.0,0.8386, 0.5954, 0.2457, 0.1303, 0.1992, 0.0732, 0.1577, 0.0769, 0.0, 0.0622, 0.2229, 0.2634, 0.3312, 0.276, 0.0517, 0.0036, 0.0, 0.0024, 0.0066, 0.0265, 0.1967, 0.388, 0.443, 0.3062,0.0828, 0.0258, 0.0, 0.0836, 0.0652, 0.0649, 0.072, 0.0454, 0.0123, 0.0, 0.0045, 0.0127, 0.0226, 0.0162, 0.0129, 0.0201, 0.0827, 0.1277, 0.1497, 0.1808, 0.1684, 0.1435, 0.1553, 0.1054, 0.0507, 0.0131, 0.0082, 0.0, 0.0137, 0.0291, 0.0952, 0.228, 0.5262, 0.8633, 1.0, 0.9473, 0.5475, 0.307, 0.0768, 0.0269, 0.0, 0.0031, 0.0831, 0.2564, 0.3284, 0.3349, 0.1559, 0.0633, 0.0297, 0.0062, 0.0136, 0.0142, 0.0063, 0.0, 0.0086, 0.0789, 0.1287, 0.1304, 0.104, 0.0312, 0.0001, 0.0044, 0.0, 0.0108, 0.0514, 0.5242, 0.8919, 1.0, 0.759, 0.1347, 0.0, 0.0299, 0.0394, 0.0438, 0.0583, 0.0983, 0.2265, 0.2884, 0.2608, 0.1576, 0.0908, 0.0426, 0.0159, 0.0203, 0.0, 0.0116, 0.0082, 0.0041, 0.0074, 0.0073, 0.0061, 0.0026, 0.0, 0.0018, 0.0289, 0.0787, 0.1273, 0.1588, 0.1438, 0.0021, 0.029, 0.0536, 0.0654, 0.6224, 0.943, 1.0, 0.7526, 0.1465, 0.0361, 0.0409, 0.0, 0.0547, 0.0035, 0.0013, 0.0078, 0.0137, 0.0091, 0.0089, 0.0058, 0.0004, 0.0019, 0.0124, 0.0078, 0.0, 0.0096, 0.0229, 0.011, 0.0125, 0.0205, 0.0183, 0.0258, 0.0352, 0.0543, 0.0586, 0.0438, 0.0743, 0.539, 0.9596, 1.0, 0.817, 0.2223, 0.0908, 0.0643, 0.0327, 0.0047, 0.0018, 0.0, 0.0012, 0.001, 0.0018, 0.006, 0.0167, 0.0183, 0.0154, 0.0109, 0.0047, 0.0017, 0.0017, 0.0032, 0.0072, 0.0084, 0.0121, 0.0176, 0.0225, 0.0244, 0.0333, 0.0378, 0.0793, 0.5985, 0.9812, 1.0, 0.7482, 0.1277, 0.0026, 0.0, 0.1135, 0.1349, 0.1357, 0.072, 0.0193, 0.0086, 0.0055, 0.0019, 0.0, 0.0002, 0.0011, 0.0013, 0.0001, 0.0033, 0.003, 0.0036, 0.0045, 0.002, 0.002, 0.0065, 0.0239, 0.1178, 0.1516, 0.1509, 0.0873, 0.0065, 0.0, 0.0036, 0.0051, 0.0092, 0.019, 0.1134, 0.3031, 0.344, 0.3183, 0.1203, 0.0039, 0.0, 0.0025, 0.0127, 0.2111, 0.5219, 0.5879, 0.9621, 1.0, 0.909, 0.7315, 0.1661, 0.0396, 0.0158, 0.0342, 0.0507, 0.047, 0.0386, 0.0192, 0.0107, 0.0077, 0.0057, 0.0, 0.0007, 0.0039, 0.0357, 0.1581, 0.1871, 0.1904, 0.0923, 0.0233, 0.0106, 0.0, 0.0042, 0.0073, 0.0103, 0.0235, 0.0, 0.0141, 0.0848, 0.2068, 0.268, 0.2189, 0.1564, 0.0329, 0.0, 0.0015, 0.0056, 0.0168, 0.0331, 0.0604, 0.204, 0.2284, 0.1526, 0.0495, 0.0067, 0.0034, 0.0077, 0.0, 0.0036, 0.0022, 0.0044, 0.008, 0.012, 0.0122, 0.0137, 0.0228, 0.0634, 0.119, 0.1267, 0.1246, 0.2179, 0.2605, 0.2266, 0.1616, 0.0, 0.0351, 0.5553, 0.9467, 1.0, 0.7691, 0.1558, 0.0305, 0.0032, 0.0, 0.1962, 0.4798, 0.5053, 0.4719, 0.123, 0.0, 0.0081, 0.3325, 0.7679, 0.7896, 0.7501, 0.328, 0.0, 0.048, 0.5529, 0.636, 0.6183, 0.3275, 0.0364, 0.0278, 0.0632, 0.0551, 0.0479, 0.0258, 0.0023, 0.0085, 0.0, 0.0096, 0.0852, 0.1798, 0.2161, 0.2437, 0.1181, 0.0756, 0.0181, 0.0017, 0.0, 0.0021, 0.0151, 0.0259, 0.1927, 0.8058, 1.0, 0.9588, 0.4359, 0.0697, 0.0296, 0.0139, 0.0, 0.0027, 0.0394, 0.0731, 0.1628, 0.1626, 0.1154, 0.0491, 0.0127, 0.0, 0.0014, 0.0159, 0.0109, 0.0118, 0.0061, 0.0031, 0.0, 0.0006, 0.0088, 0.0107, 0.0151, 0.0141, 0.0101, 0.0446, 0.0784, 0.0911, 0.086, 0.0431, 0.0, 0.039, 0.2014, 0.2844, 0.4925, 0.9865, 0.9501, 1.0, 0.515, 0.2809, 0.1417,0.0924, 0.1871, 0.1295, 0.1118, 0.046, 0.0, 0.0187, 0.1857, 0.3657, 0.3766, 0.3477, 0.5284, 0.5248, 0.5072, 0.2674, 0.091, 0.2018, 0.2106, 0.187, 0.066, 0.02, 0.0048, 0.0043, 0.0, 0.009, 0.0278, 0.0335, 0.0468, 0.0391, 0.0306, 0.0333, 0.0202, 0.0, 0.0022, 0.0172, 0.0897, 0.0906, 0.1088, 0.2429, 0.3827, 0.3202, 0.2369, 0.0169, 0.0, 0.5509, 0.9895, 1.0, 0.7898, 0.1203, 0.1277, 0.1263, 0.1254, 0.0, 0.0399, 0.177, 0.3066, 0.3053, 0.1489, 0.0923, 0.0305, 0.0176, 0.0078, 0.0199, 0.0095, 0.017, 0.014, 0.0048, 0.0041, 0.0002, 0.0, 0.0061, 0.0039,0.0009, 0.1893, 0.5694, 0.7199, 0.7306, 0.4519, 0.0475, 0.1733, 0.1656, 0.1052, 0.0294, 0.0083, 0.0003, 0.0008, 0.0005, 0.0026, 0.0, 0.001, 0.0051, 0.0114, 0.0203, 0.0136, 0.0158, 0.0046, 0.0, 0.0074, 0.0166, 0.0302, 0.0315, 0.0409, 0.0452, 0.0732, 0.3654, 0.6157, 1.0, 0.8858, 0.5435, 0.2799, 0.2274, 0.2219, 0.1655, 0.0736, 0.0809, 0.0817, 0.0637, 0.0224, 0.0025, 0.0, 0.0229, 0.0539, 0.0642, 0.0476, 0.0159, 0.0098, 0.0018, 0.0037, 0.0, 0.0036, 0.0068, 0.0282, 0.0837, 0.1527, 0.2506, 0.1078, 0.0356, 0.0572, 0.0291, 0.0, 0.0093, 0.0239, 0.0199, 0.0597, 0.081, 0.1068, 0.2952, 0.7739, 0.6121, 0.582, 0.234, 0.0004, 0.0265, 0.0562, 0.0486, 0.0211, 0.0041, 0.0, 0.0131, 0.005, 0.0205, 0.0274, 0.2091, 0.4414, 0.455, 0.3995, 0.0824, 0.0041, 0.0, 0.0162, 0.094, 0.765, 0.9919, 1.0, 0.584, 0.0784, 0.0355, 0.0148, 0.0, 0.0246, 0.052, 0.0969, 0.0562, 0.059, 0.0266, 0.0078, 0.0007, 0.0, 0.0002, 0.0188, 0.0236, 0.031, 0.0346, 0.0269, 0.0126, 0.0, 0.0211, 0.0309, 0.0278, 0.0233, 0.0114, 0.001, 0.0039, 0.0139, 0.0322, 0.4043, 0.9828, 1.0, 0.9159, 0.2014, 0.0515, 0.0293, 0.0252, 0.0149, 0.0079,0.0029, 0.0105, 0.013, 0.0, 0.0961, 0.1644, 0.1706, 0.1521, 0.0121, 0.01, 0.0232, 0.0146, 0.0167, 0.0, 0.0223, 0.0471, 0.0495, 0.0372, 0.0086, 0.0, 0.0186, 0.018, 0.0271, 0.0275, 0.0, 0.0166, 0.0226, 0.0321, 0.0393, 0.0459, 0.041, 0.3247, 0.9571, 1.0, 0.9449, 0.2964, 0.0583, 0.0421, 0.0248, 0.0075, 0.0126, 0.0094, 0.0083, 0.0131, 0.0182, 0.008, 0.011, 0.0018, 0.0011, 0.0, 0.0012, 0.0098, 0.0049, 0.0131, 0.0159, 0.0328, 0.0298, 0.025, 0.0033, 0.0029, 0.0, 0.0104, 0.0078, 0.0181, 0.021, 0.0173, 0.0169, 0.0155, 0.002, 0.0059, 0.016, 0.0277, 0.1976, 0.7394, 0.7968, 0.7784, 0.2845, 0.0213, 0.0151, 0.0, 0.0077, 0.0109, 0.0299, 0.0242, 0.0514, 0.3775, 0.584, 0.5969, 0.5197, 0.0833, 0.0154, 0.0, 0.0266, 0.1562, 0.1623, 0.1824,0.1466, 0.0772, 0.0998, 0.0762, 0.0254, 0.0223, 0.0045, 0.0029, 0.0052, 0.0, 0.0134, 0.0224, 0.1045, 0.8053, 1.0, 0.975, 0.5747, 0.0477, 0.0333, 0.0148, 0.0197, 0.0356, 0.0218, 0.0167, 0.0076, 0.0, 0.0044, 0.0016, 0.0137, 0.0074, 0.0096, 0.0058, 0.0, 0.0056, 0.0191, 0.0424, 0.0983, 0.0955, 0.0922, 0.0274, 0.0085, 0.0096, 0.0133, 0.0, 0.0568, 0.0774, 0.4794, 0.1605, 0.1373, 0.1225, 0.0477, 0.0, 0.0598, 0.0517, 0.0614, 0.0615, 0.0505, 0.0478, 0.0535, 0.0721, 0.0559, 0.0278, 0.0, 0.0091, 0.0406, 0.0409, 0.0315, 0.0168, 0.0, 0.0142, 0.0723, 0.8007, 0.9737, 1.0, 0.6146, 0.0545, 0.0283, 0.0095, 0.0177, 0.0147, 0.0252, 0.0245, 0.0158, 0.0175, 0.0, 0.0207, 0.0194, 0.0192, 0.0113, 0.0, 0.0099, 0.0088, 0.0033, 0.0086, 0.0034, 0.0146, 0.0078, 0.012, 0.009, 0.0214, 0.1742, 0.0679, 0.0651, 0.0757, 0.0, 0.0207, 0.0112, 0.0781, 0.03, 0.0504, 0.0338, 0.06, 0.0223, 0.0406, 0.0112, 0.0223, 0.034, 0.0215, 0.0266, 0.1124, 0.1456, 0.1265, 0.121, 0.1214, 0.4383, 1.0, 0.9669, 0.8901, 0.2028, 0.0, 0.0108, 0.0095, 0.0223, 0.0152, 0.0315, 0.0318, 0.0153, 0.0109, 0.0, 0.0169, 0.0173, 0.0316, 0.021, 0.0132, 0.0155, 0.0129, 0.0086, 0.0127, 0.0434, 0.0333, 0.0346, 0.0313, 0.0421, 0.0814, 0.8594, 1.0, 0.9961, 0.5917, 0.0173, 0.0, 0.0915, 0.1623, 0.135, 0.1153, 0.0381, 0.4212, 0.4975,0.4886, 0.2634, 0.0568, 0.0248, 0.019, 0.0195, 0.0049, 0.0142, 0.0028, 0.0, 0.0049, 0.0085, 0.0139, 0.0116, 0.0121, 0.0027, 0.001, 0.0, 0.0205, 0.035, 0.0304, 0.0338, 0.0192, 0.0076, 0.0073, 0.0129, 0.0088, 0.0012, 0.0044, 0.0151, 0.0273, 0.0423, 0.2313, 0.9567, 1.0, 0.9654, 0.3573, 0.0742, 0.0757, 0.0606, 0.0506, 0.0148, 0.0061, 0.0081, 0.0061, 0.0116, 0.011,0.0065, 0.0134, 0.0077, 0.0046, 0.0083, 0.0, 0.0027, 0.002, 0.0035, 0.0066, 0.005, 0.0045, 0.002, 0.0118, 0.0055, 0.0144, 0.0107, 0.0069, 0.0185, 0.0061, 0.0123, 0.0218, 0.0059, 0.0063, 0.0, 0.0015, 0.0149, 0.0088, 0.0146, 0.0241, 0.0046, 0.0175, 0.0246, 0.0285, 0.0405, 0.5273, 0.9873, 1.0, 0.8664, 0.1606, 0.0337, 0.0224, 0.0362, 0.0034, 0.0148, 0.0166, 0.0, 0.0078, 0.0173, 0.0087, 0.0312, 0.023, 0.0122, 0.0181, 0.0, 0.0114, 0.0247, 0.0185, 0.0127, 0.0311, 0.0132, 0.011, 0.0, 0.0218, 0.027, 0.0669, 0.0406, 0.0643, 0.0309, 0.0334, 0.0627, 0.0258, 0.05, 0.0752, 0.1124, 0.072, 0.0314, 0.0, 0.0265, 0.0314, 0.0369, 0.0316, 0.0432, 0.0299, 0.0313, 0.1602, 0.8333, 1.0, 0.9239, 0.5185, 0.1019, 0.0456, 0.0635, 0.0, 0.0406, 0.0221, 0.0187, 0.0579, 0.022, 0.0233, 0.0189, 0.0058, 0.0026, 0.0019, 0.0107, 0.0069, 0.0019, 0.0038, 0.0234, 0.0285, 0.0347, 0.0291, 0.018, 0.0038, 0.0097, 0.0194, 0.0193, 0.0251, 0.0104, 0.0015, 0.0422, 0.093, 0.0991, 0.1117, 0.3346, 0.9415, 0.9733, 1.0, 0.2816, 0.0534, 0.0216, 0.0, 0.004, 0.0229, 0.0489, 0.1233, 0.2401, 0.2256, 0.179, 0.0708, 0.0158, 0.0055, 0.0164, 0.0391, 0.0367, 0.0535, 0.0219, 0.0075, 0.0039, 0.0, 0.002, 0.0099, 0.0132, 0.0276, 0.0935, 0.2293, 0.2528, 0.2092, 0.0795, 0.0074, 0.0003, 0.0332, 0.0208, 0.0001, 0.0382, 0.012, 0.1493, 0.1071, 0.0829, 0.0, 0.0932, 0.0348, 0.2302, 0.2609, 0.7561, 0.7316, 0.7616, 0.6383, 0.324, 0.1929, 0.1186, 0.1691, 0.0484, 0.032, 0.0, 0.0988, 0.0889, 0.102, 0.2644, 0.7656, 1.0, 0.9521, 0.6169, 0.0864, 0.0766, 0.1902, 0.157, 0.0984, 0.0705, 0.0, 0.0143, 0.0795, 0.0855, 0.107, 0.0643, 0.0273, 0.0031, 0.0044, 0.0236, 0.0161, 0.0, 0.0062, 0.0237, 0.0167, 0.0434, 0.0323, 0.0165, 0.0463, 0.1735, 0.3714, 0.3851, 0.3441, 0.1225, 0.0264, 0.0, 0.0096, 0.0234, 0.1553, 0.8602, 0.9595, 1.0, 0.475, 0.0586, 0.0166, 0.0, 0.0064, 0.0185, 0.0568, 0.0784, 0.0919, 0.0453, 0.0367, 0.0107, 0.0005, 0.0, 0.02, 0.0212, 0.0089, 0.0031, 0.004, 0.0183, 0.0233, 0.0164, 0.0183, 0.0431, 0.0078, 0.0, 0.1682, 0.2101, 0.1578, 0.1449, 0.2036, 0.0309, 0.1029, 0.0845, 0.1036, 0.0948, 0.0, 0.0472, 0.0409, 0.0591, 0.0914, 0.0342, 0.0114, 0.0, 0.0527, 0.0062, 0.0762, 0.1126, 0.1299, 0.0824, 0.2079, 0.357, 0.4242, 0.4334, 0.1262, 0.0374, 0.0, 0.0398, 0.0052, 0.0618, 0.7568, 1.0, 0.9866, 0.7159, 0.0553, 0.0129, 0.0512, 0.0149, 0.0194, 0.0402, 0.0116, 0.0185, 0.0116, 0.0, 0.0217, 0.0162, 0.0165, 0.0158, 0.0442, 0.074, 0.0502, 0.0385, 0.038, 0.0428, 0.0332, 0.0298, 0.0273, 0.0, 0.0158, 0.0875, 0.1455, 0.0908, 0.0418, 0.0875, 0.0184, 0.0, 0.0232, 0.0031, 0.1351, 0.3651, 0.4047, 0.3856,0.1322, 0.0582, 0.0, 0.144, 0.1574, 0.1898, 0.1904, 0.0873, 0.0044, 0.0, 0.2353, 0.3982, 0.5428, 1.0, 0.6488, 0.6549, 0.1882, 0.027, 0.0236, 0.0287, 0.0372, 0.024, 0.0, 0.0286, 0.0521, 0.1146, 0.1733, 0.1138, 0.0995, 0.107, 0.0113, 0.017, 0.0, 0.0857, 0.0918, 0.0357, 0.0797, 0.0839, 0.1096, 0.0768, 0.057, 0.0791, 0.0, 0.1551, 0.287, 0.2965, 0.2604, 0.303, 0.1853, 0.1583, 0.166, 0.0378, 0.0137, 0.0139, 0.0208, 0.0, 0.0119, 0.0195, 0.0324, 0.0203, 0.0209, 0.0153, 0.0014, 0.0, 0.0036, 0.0248, 0.0112, 0.016, 0.0147, 0.0165, 0.0341, 0.3013, 0.9218, 1.0, 0.9564, 0.3257, 0.0216, 0.0199, 0.0145, 0.0063, 0.0, 0.0735, 0.1208, 0.1093, 0.0623, 0.0242, 0.0, 0.0292, 0.0231, 0.0261, 0.037, 0.0882, 0.081, 0.0797, 0.0256, 0.0178, 0.0129, 0.0, 0.0199, 0.0047, 0.0334, 0.1402, 0.083, 0.1392, 0.0293, 0.0, 0.0519, 0.1237, 0.2078, 0.334, 0.963, 0.4676, 0.3278, 0.1776, 0.1245, 0.1934, 0.2072, 0.1503, 0.0161, 0.0, 0.0035, 0.0567, 0.3051, 0.3403, 0.331, 0.1751, 0.0489, 0.0433, 0.0347, 0.0, 0.0427, 0.0283, 0.0667, 0.0718, 0.1266, 0.8245, 0.9794, 0.9698, 0.6504, 0.1564, 0.7114, 0.9146, 1.0, 0.7494, 0.4129, 0.285, 0.2439, 0.0723, 0.0556, 0.0529, 0.0, 0.0141, 0.038, 0.0439, 0.1313, 0.1417, 0.1012, 0.0378, 0.0188, 0.0, 0.0397, 0.0355, 0.0234, 0.0234, 0.0642, 0.0337, 0.4741, 0.6511, 0.6066, 0.4453, 0.0107, 0.0, 0.0061, 0.0085, 0.0162, 0.0348, 0.0238, 0.03, 0.0317, 0.0075, 0.0, 0.0112, 0.0076, 0.0211, 0.0167, 0.0338, 0.1149, 0.4, 0.6236, 0.9503, 0.7554, 0.9219, 1.0, 0.9178, 0.8877, 0.2196, 0.0658, 0.0362, 0.0, 0.025, 0.0353, 0.0502, 0.0306, 0.0128, 0.0, 0.0107, 0.0056, 0.2028, 0.3142, 0.3082, 0.2825, 0.0473, 0.0193, 0.0, 0.0539, 0.1227, 0.1103, 0.0951, 0.0162, 0.0, 0.0154, 0.0305, 0.1771, 0.2153, 0.1999, 0.0575, 0.0696, 0.0637, 0.0509, 0.0436, 0.0, 0.0316, 0.1027, 0.1059, 0.0805, 0.0382, 0.0, 0.059, 0.1714, 0.1712, 0.2362, 0.1503, 0.1068, 0.0, 0.0129, 0.0048, 0.5637, 0.7463, 1.0, 0.7426, 0.4598, 0.2528, 0.0415, 0.0848, 0.013, 0.0668, 0.0253, 0.0, 0.129, 0.0582, 0.2155, 0.2087, 0.2984, 0.2151, 0.1958, 0.0, 0.2347, 0.7849, 0.8233, 0.8825, 0.4161, 0.0971, 0.1023, 0.0, 0.0239, 0.0547, 0.0965, 0.062, 0.009, 0.0488, 0.1995, 0.2373, 0.2043, 0.2035, 0.1408};
//// Max filter + Gaussian
//    vector<double> temp2 = {0.19222672956329534,0.2746050645667184,0.4025547100586064,0.5148805955589876,0.5551109459481904,0.504198924228711,0.38996710928893813,0.26396603663120144,0.16700527471634583,0.11146032106738182,0.08818089468895646,0.08443878019000427,0.09678641770033512,0.13338944791993187,0.2076177329216307,0.32497277181998546,0.46892223658960963,0.5989754892294663,0.6697258726502671,0.6586600130757314,0.5782032135306935,0.4628917056399792,0.34627394776486164,0.24700966402394478,0.16975275937287299,0.11266372138289192,0.07275256366476755,0.04763070998160545,0.036270653210181844,0.03941233675622374,0.05811384639513503,0.09028465743796499,0.1279313189276079,0.15826506603806878,0.1693518321251309,0.1575406471217024,0.13119742438984466,0.10628794421314414,0.09594669877400809,0.10239197137201442,0.11744537104193803,0.12983578970430945,0.1326777755322663,0.12662991987536842,0.14474309201608726,0.18903544761610372,0.2310362154168464,0.2909570682889467,0.3575532858976141,0.3991400063580106,0.3932530797313677,0.3406019872793493,0.2635590664724287,0.19180107298939544,0.14738637467649754,0.138596077123358,0.160531416806933,0.19747574042032873,0.22815909954632455,0.23515251166859982,0.2136357136853634,0.17385787997338317,0.13608608768227537,0.12235577444019859,0.15055902604138563,0.23022383701815582,0.3553176753574767,0.49751147945849755,0.6109462055683271,0.6520517328499666,0.6036777917394782,0.48639536185877996,0.34676111246979724,0.23049175458539276,0.16084092233634537,0.13437406355835374,0.13097266809101465,0.1297340618424723,0.12203459883298765,0.1133477785283335,0.11290115252876354,0.12152980911431424,0.12943558465157595,0.12517719523665813,0.10620541553939741,0.08048975238960457,0.05859102055427107,0.0455164811296745,0.0400805286043402,0.03956008303074482,0.04312388998671399,0.05180405903252107,0.06625181308090686,0.08403120864847236,0.09923824625725416,0.11523604011059989,0.22714136888313027,0.26810122008022025,0.24782978580286835,0.24793365593736647,0.27817216186961535,0.3367229690505968,0.4140842402018596,0.5007870219902575,0.5891190631624376,0.6675135578126447,0.7163149421790219,0.7134371422268421,0.650150573851475,0.5450853696599415,0.4398919596003537,0.3331354100011805,0.2464179137213157,0.26306640944576715,0.25492456867196134,0.2521859550560203,0.2305197623694905,0.1990122374580853,0.1692669381611597,0.14971776882329182,0.14283535661426724,0.14516247180318612,0.15134545341261144,0.1577673365117229,0.16112740297091513,0.1565623652237031,0.12155570680937092,0.0490858979818826,0.018134902168829937,0.011451559613517358,0.013696316489511568,0.01770818497429946,0.023233834851574965,0.030857372976647632,0.040681402720477954,0.05118848680684687,0.05933529627632045,0.06217682298939295,0.05882496089412898,0.05115103040810701,0.04246237738542933,0.03509133850634463,0.029205841949133393,0.024004956365504267,0.019705133350888526,0.018240057417159086,0.022844948846256208,0.03832024677258915,0.07251742512911294,0.13654346466274714,0.23937793060055837,0.3759985184010511,0.5172077928244873,0.6152900787484302,0.6294900576436246,0.5535114724477485,0.41994774711659405,0.277499229049724,0.16277196638685296,0.08901142682679174,0.05297492245270937,0.04612485296404955,0.059727350899890196,0.08409306216407438,0.10798866553610152,0.1228384752634529,0.13037925961405247,0.14634229992941264,0.19292015939238408,0.28231585742483756,0.4042565354841462,0.5286887297337854,0.620698768696669,0.6550060394376669,0.6226440662910129,0.531753303266857,0.4052742663042076,0.27441441520063625,0.16803053763824402,0.10349977539326563,0.0999259069711103,0.12210268400481217,0.17182587649439374,0.2283468991828166,0.2750176672233172,0.3060679000954693,0.3251668122229075,0.33502197729099603,0.3294346015681346,0.29876308167709137,0.24302337155884418,0.17618320630091977,0.11648114781662976,0.07453469080829106,0.05105415062522544,0.04261236273970861,0.04631379854192878,0.05896998124499411,0.07447641034725025,0.09644194302756247,0.10550413934900905,0.15615601084699082,0.13907591931625077,0.14155320035515123,0.17961321132503968,0.25210333140381863,0.3385568101068719,0.40554602058967126,0.42378969974222874,0.38737425134316844,0.3189060709203342,0.25290182123966953,0.21050125359190808,0.18818865620331404,0.18440687104473483,0.20495668651970272,0.15294449791168432,0.10830604542762695,0.08464691980354905,0.08293534057179061,0.09533533009428889,0.11530387514188252,0.14327637778127297,0.18197914497739215,0.22634147017441203,0.26028096214245416,0.26614166300674436,0.23869951002175271,0.19064361743535424,0.14429437743187645,0.11628215836745685,0.10800672307730226,0.10884882715266644,0.10679260575161084,0.12908230831009976,0.19434434824435812,0.15198243319829619,0.1191328537487334,0.1036735424968877,0.11311518918996308,0.15204983059101665,0.21418890117262288,0.19195126347873176,0.1791302479899106,0.1714146213390344,0.18545212880919937,0.19139044747568254,0.1860238422624327,0.10335707347152058,0.08592780990159665,0.09274056985441273,0.10982750721538226,0.12802336740876877,0.13582140507192003,0.12861030009585292,0.11177014653663843,0.09558531184890923,0.08712502659474813,0.08587849639923067,0.08592772655403032,0.08212307353792075,0.07549183227488802,0.07438459979165818,0.09025794158191211,0.130082622222761,0.1894505281520529,0.25141629508832325,0.2960303755803802,0.31866372370629137,0.34081677875029676,0.3948515888917292,0.48915520913664706,0.588577848159569,0.6364609261530368,0.6025056145458023,0.5119555693835148,0.4272897259083959,0.3980451671056056,0.4240179586677696,0.46197587669410006,0.46385311123531364,0.41042591557283953,0.3172526676818427,0.21754468279801928,0.14198694478354776,0.1097616361914523,0.12874275672997976,0.19419572542314456,0.2836888265576244,0.3590993008236337,0.3838755268960889,0.34561919174168765,0.2643099248661124,0.17826582237552943,0.11424427414962644,0.05409914472512914,0.04889628083811103,0.05398637395058819,0.05481763251722007,0.049775407410190986,0.04162724526383364,0.03411106180003992,0.029655176914815815,0.030123076257841223,0.0388910096140728,0.06035926184069327,0.095260892234709,0.13561220171068467,0.16585211064664643,0.17194908449094334,0.1519294040272095,0.11845816628046871,0.09092169887859354,0.08438698921513693,0.10548504510701208,0.15708635033640297,0.24264050935672812,0.36014227388789777,0.48886341813777423,0.5865942392263824,0.6104304332332754,0.5486975655810793,0.4338796185113777,0.320407376001433,0.24556763035421383,0.20928489107081646,0.18873362917472994,0.16714729301677,0.14485113484471893,0.127392281373643,0.11315022936447965,0.0957185219406898,0.07294378731019584,0.04933090453286773,0.03083857681469531,0.019909394085904395,0.015567735171557151,0.016542121218622564,0.023123935975949732,0.03599929343298872,0.05376585899654008,0.07193728602755176,0.08484326748554469,0.08902147459079857,0.08539195994842572,0.07928019390907577,0.09853891255934022,0.18938558415979795,0.27029320699335385,0.391375402010048,0.5132849669658655,0.5794993009915916,0.5527470848486559,0.44277576715114986,0.29872752859743645,0.1741525346882966,0.09724722972369473,0.06678507200573705,0.06605735089824892,0.07687161419989794,0.08582940432405314,0.08591818217341475,0.07724323944118122,0.06635246819647368,0.06216373279347603,0.06965165723722203,0.08612158837452415,0.10375739028506471,0.11599774553961545,0.12099369546717734,0.11942060440091007,0.11145512692716242,0.09806092260577587,0.08449626255898873,0.07959167865008294,0.08906105508087006,0.10898142567041942,0.1364261511724462,0.18966041196290254,0.17150491350207905,0.14153622381812062,0.11592895043951235,0.10261107254159955,0.09957287310170626,0.10400096153417412,0.12162860039908109,0.166760347049068,0.2515680122092735,0.37072107963418105,0.4931169090142749,0.5719538553251104,0.571367320774752,0.49127811514543834,0.3699649608265745,0.2612979960495794,0.20487980381577672,0.21191609414072698,0.2719664785010216,0.3119704524249123,0.31882578730822403,0.23976256062270784,0.2118014529734503,0.22336352473714227,0.21862984168248004,0.19896578024312547,0.16876214366463588,0.1338298853929971,0.09987045008399578,0.07119401558882255,0.049823943941910746,0.035508786198300696,0.026897924750513013,0.02315482472371032,0.024536512568893495,0.03143497057062088,0.04322532303641294,0.05869345497366442,0.07802784040192587,0.10471821290469151,0.14573556983451003,0.20926606451399654,0.30043263232865175,0.4161298061716927,0.5401065057936748,0.6414719521794258,0.683208426854903,0.6420776312738867,0.5267153396876072,0.3757989830702747,0.23513735423202686,0.13342776271572787,0.07534341415606399,0.050528349335315755,0.04473522039117415,0.045568470893510536,0.04479625525584587,0.03968836565628072,0.032559052097799863,0.027917857267024865,0.02899675144923147,0.035784418977173695,0.045073785419046816,0.05204722554822055,0.05320601286472814,0.04952334351625069,0.046863573393936826,0.05201267839039668,0.06697962763099298,0.08639512525443668,0.10038372595734232,0.10105200173535875,0.09444924757815112,0.10609011704081796,0.2679994684036162,0.265738121756775,0.2376623406313415,0.266063965764335,0.29705988847355874,0.31202724420114997,0.31531270102674336,0.3132262666670798,0.3030553594058812,0.2777752350383392,0.23947066723652136,0.2031505861057391,0.1838765380298796,0.1813321104973783,0.18023602933526853,0.16590197601418874,0.13760644696073646,0.10788013803990248,0.09200782702255282,0.09905036508597309,0.1300495791628632,0.18145140568318685,0.25030657170288395,0.33803715822794683,0.4465028219236472,0.5647766488895165,0.661056683049558,0.6961467694921627,0.6509265644103676,0.5414455081319224,0.40782461692198413,0.2887029340366437,0.20282776005374328,0.14885570581492574,0.11781741129069355,0.10486787402614349,0.11117213551244959,0.1365355656260172,0.17153641991380159,0.1981647364065009,0.19964907366629916,0.17156125607508274,0.1251889851435228,0.08086805118106634,0.058073675362146665,0.05793948499601006,0.08775165791655101,0.1265235928294121,0.18905992187990267,0.2369497218403649,0.24883309626154468,0.2201982162482416,0.16715377977099916,0.11456914488539129,0.07952908712623258,0.06412924149819882,0.060391748118060286,0.05922642543119292,0.055697743676372015,0.03942422019382337,0.015338971910370787,0.00941448792040265,0.009636262964195506,0.012576322667579765,0.017080123562940263,0.024430380860740782,0.037016252973595336,0.056552552841436614,0.08188741345567677,0.10860372311508752,0.13094101096526578,0.14432809383130174,0.1466403808064336,0.13788150328901044,0.11949268687019171,0.09443107193433443,0.06754213161154622,0.044909167397124146,0.032523072161773796,0.03618486089527356,0.0631104164718557,0.12268675657383916,0.22261310690386477,0.35968286313267345,0.5108400094874864,0.6346054313417782,0.687917048160412,0.6498606211602816,0.534715318153631,0.383778755609356,0.24310158681425426,0.14436119968238625,0.09901360074186884,0.10202198381972413,0.1367293295111522,0.1789619518993594,0.204294420978939,0.19856713438896112,0.16432288262716205,0.11682235984487051,0.07280511396489749,0.041582879575641926,0.02403078657183589,0.01715449894806153,0.018704263513448883,0.028279663794439854,0.044922353452070456,0.06421604578530049,0.07852229597238829,0.0810886201160203,0.07069601238012309,0.05295866295930948,0.040024928193451666,0.05118827750276001,0.08610252765599336,0.1697667756510963,0.29985703050003165,0.4460786733454581,0.5541384489572957,0.5747296710816283,0.4980642523369205,0.3620537390945302,0.22503977051647753,0.12882488806052927,0.08373562154765536,0.07950894678732386,0.1014287374716157,0.13602584458189537,0.16919144479412948,0.18684201151574623,0.1806942156288896,0.15291144841057888,0.11405548168039494,0.07609964478403715,0.04658088499235874,0.02730148844380938,0.01639167685765777,0.010905183248711906,0.008355482982436819,0.007144035393118021,0.006403790875086978,0.00574201137810575,0.005193457302560815,0.005322984551876458,0.0071351098022022635,0.015195155164893524,0.05110365839476204,0.07306720009965315,0.09031517735098993,0.09692844678304564,0.09510440543385543,0.09903159262299649,0.1316518818321998,0.212787230080951,0.34184573509842936,0.4858607280366507,0.5881915335755911,0.6005715752007159,0.5161717915763261,0.37412527910779764,0.230892440782405,0.1250520122900296,0.06413827722061764,0.03582668348700669,0.01534769090729136,0.007686408304278636,0.007603389458242315,0.008007672860533646,0.007933721827644924,0.007231075948704018,0.0062943777397159485,0.005697651909689712,0.005740444476072455,0.006347645147350658,0.007376838896349982,0.008839548377610933,0.010704444219610337,0.012726023454969487,0.014734151230728836,0.016996895445341807,0.020160488920945945,0.024948316731439116,0.03226442947445059,0.044436843850533356,0.06801429095645413,0.11607669163823918,0.20393909620635386,0.3343411205283126,0.48130274014546687,0.592641881292038,0.6189150419452786,0.5479111874173993,0.41313071359286224,0.2685404160791533,0.15422042635234343,0.08160419299861826,0.010345510658494754,0.003449484660398143,0.0022735708449542697,0.0023299511564555133,0.0034488227533909337,0.005551181367013974,0.008228763548972232,0.010647213390684807,0.011877029707779022,0.011451836663449748,0.009672921869569516,0.007419436669594673,0.005661478316048477,0.0050385529069917594,0.005724985183356416,0.00757282201576304,0.010382623089587416,0.014183534631234498,0.01984237276980465,0.03064542701132371,0.054920688467514314,0.10719451528784636,0.20216904725280313,0.33904003586844983,0.48667579845911296,0.5895160333191021,0.5998122479269965,0.5126871994412838,0.3714727985713855,0.23701957112006938,0.14854491998980282,0.10806741671001684,0.09388232370372311,0.08363171688333068,0.06747050106087155,0.04716546073208803,0.028479152452663074,0.015100934894637996,0.007262993116124001,0.00337175206901492,0.0017406553878154122,0.0012654434117878668,0.0013646199714866267,0.0017553280968029949,0.0022768119362207894,0.002838269752823971,0.003568525425067995,0.005174300556597955,0.00932804273130223,0.01851426005908947,0.03458002837971092,0.0561871132196444,0.07966416274722124,0.09206851171867848,0.08917061811188239,0.07238492096487445,0.049741413540761684,0.030863663729009384,0.022706509745505917,0.029359009072347585,0.05355008684739777,0.09514238946033242,0.1464288267772875,0.19047402108029488,0.20824031864793752,0.19123425672692607,0.14904984708450372,0.10515302216218825,0.08556960312789878,0.10961704490787585,0.18560812478634545,0.30863651350615745,0.4594155880728963,0.6061834719882849,0.7108803047615345,0.7401848738688132,0.6801128213206039,0.5467046784812654,0.382003543560354,0.23335363655489857,0.12977094891215762,0.07330225140921592,0.04859573953106076,0.038158640024293526,0.03119376501707825,0.024106280463270964,0.017232300123472905,0.011963128573689091,0.009810202448209132,0.012895008261289538,0.024075412744448483,0.04517160509745748,0.07367259628723756,0.10109525587830846,0.11626516749566675,0.11229376855876988,0.09139354410904767,0.06304730193250274,0.03742863594173449,0.019987445893610494,0.010733509845521958,0.007589776620464206,0.007990573044347077,0.03400938973605389,0.04761806690905652,0.07402428842033844,0.10984623217575175,0.1436269692187335,0.16096280009787875,0.15346216347277172,0.12427392501823803,0.08581709515084758,0.05214818614870683,0.03235429474658021,0.029278156465233574,0.042155619591052365,0.06837627370841443,0.10207014289804896,0.1325898842956532,0.14750644056995785,0.09505782709715214,0.044957848930693906,0.029401375633235683,0.017793580682276832,0.010014202728281657,0.00598558864952062,0.004614927414913858,0.004930720521646299,0.006402904633263533,0.008923324075572447,0.012974583979072488,0.019877233791789992,0.03151162547837498,0.04920247413872337,0.07260635555690233,0.09987627232319013,0.12863520627815647,0.15572709744188876,0.17613719440060685,0.1852657479437016,0.18667513389344817,0.19882566476563304,0.24881479294419775,0.3497619412677454,0.47830299371952734,0.5780190950636865,0.5946100691789536,0.5157313457039213,0.3807400600738864,0.2535889882251881,0.18407067335093935,0.1853758539032129,0.23481274934605864,0.29005984181152,0.31328326130830014,0.2940382262528082,0.2571072987505596,0.2461427464727823,0.29108499421969436,0.3825548658525604,0.47443381052312456,0.5153732522672609,0.48664830013020405,0.4168036361190707,0.35948345781042607,0.3501878146383951,0.37833278575690865,0.3996683410473671,0.3759107000801262,0.30388766947122764,0.21140620704934027,0.13084079149692665,0.09832975828770413,0.06150701217417161,0.041911953469158277,0.02993357991936592,0.023280108273834522,0.024496697939399112,0.03727376302565617,0.06288555624653942,0.09711675034366057,0.12971914698857892,0.14835307801765885,0.1453513588797797,0.12228637398599539,0.08865292833048659,0.056157882521477506,0.03566597844469956,0.03480035248374792,0.056170632481468856,0.11691692719363836,0.2228486871914274,0.35968044972817875,0.4995775267080349,0.607220581641155,0.5744302981431681,0.45582950049078497,0.3043055555337712,0.17285446591885847,0.08800421055789875,0.04939639987838508,0.04429631027128975,0.058764859085847776,0.08013233608501258,0.0966545117343638,0.09974839292468841,0.08757148363262662,0.06557045288183895,0.04250285964115518,0.023974833039819825,0.014649530019213082,0.010642878164685572,0.009241758021078381,0.008129869652423436,0.0066921971535169765,0.005377522351489017,0.004925497239794483,0.005767363332981635,0.007864156611068152,0.011072121258045627,0.015756768901372804,0.022918734742631876,0.03322446494392755,0.04547642974671936,0.056329058297777755,0.06265203905138657,0.06563117028871304,0.07359020437283463,0.10059672684713841,0.16099059150220513,0.2623713005880599,0.39884405114057464,0.5459182198028731,0.6628385162942513,0.7086390644198671,0.6657965118133646,0.552870827503326,0.4136752926627158,0.2910129843574805,0.20566366936374683,0.15419926776122667,0.12227896345613325,0.10018727406474986,0.08978563202285826,0.10042998156503757,0.1386974812644426,0.2004952301780276,0.27233272262106306,0.33935693828080354,0.39051286112467215,0.4169064617472231,0.4109724189909564,0.3724235121258335,0.3136576953491154,0.25396739238774474,0.20578619040548193,0.1678809837050845,0.13231726891506315,0.09550363409544649,0.06121428756809221,0.030637513029012475,0.018041034053870836,0.014643270065807668,0.017404368514903398,0.02314857245587694,0.028930213985495914,0.03255283736518525,0.032985022777990904,0.030424288134763576,0.02618339509909198,0.022724528609637897,0.023525709114497,0.032143541848643604,0.050919619362391824,0.08061637165084355,0.12046661523492408,0.16638747671149556,0.2085052951837445,0.23435352510614635,0.24081958058915456,0.24554218818885915,0.2813369388369581,0.36929781956611546,0.4908845639092823,0.5898208637919885,0.6101183221286999,0.538171008830864,0.40988865544838887,0.27936666733873977,0.18312181744351194,0.13061160549011386,0.1165361053044619,0.13099617423823345,0.15902191204797278,0.18077582802298256,0.18027209406768274,0.15492674895571634,0.11529019150685361,0.07582926620857806,0.04603511821809585,0.02797387560323938,0.01887925157064339,0.014716255084464799,0.012341408077465421,0.01012518795802341,0.00774058885133863,0.005677990715643349,0.004859778145426019,0.006688130819591021,0.013542773305198642,0.02875427628782514,0.06912132771513303,0.2887881578424034,0.39623502816271683,0.4604788156801701,0.4570360756124569,0.39471465755385343,0.3080285126459168,0.23044569369242507,0.1618409883819003,0.07294792578331803,0.032422588222658665,0.021020091567099997,0.01191430830307136,0.006031195669633713,0.003124668545448315,0.0023337881441261327,0.0029719532979461606,0.004709092834497434,0.00720267486367818,0.009762117554898512,0.011480465389099694,0.011788884723821733,0.010969793618507561,0.010179463794297093,0.010885785093804595,0.014106355371430145,0.020298712984200707,0.030618690634526078,0.049496841571965294,0.08674682046930554,0.15557529322394026,0.2635087508795707,0.3995120418782413,0.528889032955785,0.6067665473060444,0.6052710971160773,0.5323210133853734,0.42395315141382006,0.31789776030821587,0.23365337092486016,0.17261031730237814,0.12902742160016464,0.09740690646335001,0.07348204829675303,0.054206242008692396,0.03890089472585538,0.029201392287335846,0.02660861168141593,0.029856164464414626,0.034728894361144125,0.03655122137135531,0.033133319556917534,0.025699429863897585,0.017348818964683658,0.010858699446261279,0.007737595759862135,0.008682727308307225,0.014141569043038135,0.023882130973259412,0.035877961466774966,0.05201765787361959,0.0923782924559683,0.16642498058931826,0.14864546314429694,0.10462642016420452,0.0673992357627277,0.04232708554394085,0.02916493936311339,0.026531415324599173,0.034641024675193276,0.05714125613655212,0.10299045463898711,0.18487982958537902,0.3087917700850158,0.45737914493900056,0.4635795138766679,0.38815133723850914,0.3534309767535099,0.2748940382567087,0.18396507913938182,0.1106635708817458,0.06502750970286097,0.04054232507435748,0.027010315647592466,0.018979207074328673,0.01633138896941836,0.02219089607782594,0.04198802076400345,0.08165757049698205,0.14183440173496012,0.21062095986445936,0.26377781079153556,0.27749780848383315,0.24614815608188015,0.19003553998969958,0.14682249952128787,0.15383542685650983,0.23055217848258733,0.36478038592042134,0.5089289322690016,0.5982420090122588,0.5883085265723048,0.4841981468653111,0.3346010267468946,0.19672121821434085,0.10406103973067403,0.05982588265038437,0.04999392845715088,0.057390683749339824,0.06802276238358893,0.039263078301705855,0.03564442040733921,0.02828970253801939,0.019563830452706422,0.01280699730760964,0.010039201247294715,0.01138503176112967,0.015429920129118434,0.020009695297488198,0.023097636802933922,0.023618019090952166,0.021962710767142807,0.019776169713195672,0.01878421798530047,0.01938884700646442,0.020393278397212482,0.02029912122542695,0.019270620904485756,0.02099344322066153,0.03446500170291173,0.07485198938261856,0.15898796984538996,0.29144413573614736,0.44770578077212125,0.5737485933358968,0.6138671015651496,0.5492211208084289,0.4119791611268702,0.26087507434291307,0.14204912357560026,0.069501040528937,0.033473311272687864,0.018730258924149236,0.015450142157512552,0.02050900587761797,0.034273298980195516,0.056280734270671394,0.08159364760706518,0.10084092393787196,0.1052612129382963,0.09304895173878393,0.06103011961256682,0.03995243114375928,0.02682296293091296,0.020102293821613645,0.01859172694215896,0.02086440541956846,0.025207513037528873,0.029039469738526755,0.029832162053013794,0.026976127500057723,0.022343610969923412,0.018679664029150063,0.017378179467894335,0.017784347439472083,0.018408784327278202,0.018626818775263662,0.019308338082507086,0.022250024494612265,0.030088149164682595,0.04834232170180878,0.08850231955836159,0.16638437367389114,0.28982789359227573,0.44046603698464476,0.5689629620896235,0.6189174523588236,0.5656887257786632,0.4349878923660632,0.2830427283426746,0.1583020566865298,0.0789872004881447,0.03809780463188336,0.020418677478752933,0.013933823696794351,0.012030717622194506,0.01154144842680218,0.010935374943643739,0.009582616478174636,0.007580326577525987,0.005514747305764838,0.004097423029256086,0.003832092010049686,0.004866629498345646,0.007090857135770658,0.010306160634762368,0.014159391542325494,0.01784477407789342,0.02007666111064554,0.01973181633205225,0.01675398804763767,0.01240140555828041,0.008977128894904322,0.012913431255544349,0.012265001947050554,0.011626286465750063,0.013600852378235153,0.015184367425974543,0.015560578474317986,0.015262841307592425,0.01672373038779396,0.025753857888378705,0.05295688850152581,0.11177558026987273,0.20919238543835528,0.3316986512417209,0.4404329906068717,0.48852593679178197,0.4524053951545945,0.34960745925733505,0.22566399568217668,0.12306530152703328,0.060695466792984744,0.036941410509627626,0.04612762080525263,0.09104652652861384,0.18122781223539017,0.3166749207443451,0.34802142248147694,0.3491210674981618,0.3644403199851368,0.3191613672095297,0.23721308107182015,0.1574604202590417,0.10888677099665932,0.09708321570685297,0.10849858104767796,0.12394212357113601,0.13008332286445937,0.12339915968755763,0.10731142142141656,0.08693157925928824,0.0658857323775098,0.04637332103260575,0.030188204654719476,0.018945080519326574,0.0144586802235074,0.020761205285134053,0.04733420449359881,0.10946246670895991,0.21940248041217025,0.368495137856388,0.5155037763839074,0.6010500582339209,0.5852032456765311,0.4767388788179826,0.32622884991569895,0.19004293110574455,0.0984141584091687,0.05047767277466304,0.029878637203983618,0.021054015515463133,0.015658493424131704,0.011273934323758898,0.008092839753452171,0.006549534674993654,0.006410582716434071,0.006905230702336151,0.007322473177368128,0.007650644717698613,0.008885864487655945,0.012806369166849735,0.021078340346076353,0.033823697678724736,0.04842179092522707,0.05984539441618648,0.06314243287285612,0.05665400490158444,0.043251176029082614,0.02911792817269356,0.02933469437230117,0.11285108192440524,0.119814781401721,0.19230203706624102,0.3204137559335295,0.47294495630816835,0.10113251101499765,0.08995963188360738,0.08243022483092684,0.07034897399668247,0.058933081859012915,0.052964584912761806,0.05218710997514518,0.05354838408303422,0.05460097313721257,0.05477979761244173,0.054347719926296764,0.052875279734404194,0.04910800935691136,0.04249674416075027,0.034630946671092365,0.02851700364204941,0.026068949197591156,0.02660223230934574,0.0287126603155526,0.034883788252960964,0.055858154387726934,0.11015351212078135,0.21356598809734273,0.35996938083327573,0.5089482797359811,0.5999031363925418,0.5899475402122272,0.484681420417894,0.3333420959412984,0.19352535663477832,0.0979662849781085,0.0478246176119019,0.027515446420364877,0.02104456435528272,0.01885819292851053,0.01726811830850456,0.015816715305885586,0.014975955398777817,0.014748807766018156,0.014434538882321876,0.01332977820722016,0.011460104907552247,0.009491077679730344,0.008047618616461649,0.0073028295087612626,0.0071470515855107134,0.0074688861903014,0.008167387619356588,0.009078311267423655,0.010062238529535297,0.011141508475477213,0.01241685385097388,0.022456420753189436,0.1317735712343489,0.05851830415937398,0.05464965185295923,0.04724182593299457,0.03961230894575048,0.03530470107130737,0.03548286545300947,0.038347042137477236,0.041123030611031015,0.04218763623858089,0.04129656763685983,0.038742042328373506,0.03505654095404015,0.03127533708024905,0.028933942953628354,0.02977432093104217,0.03558099022762438,0.04784282698564889,0.06677020214353327,0.09128454618383414,0.12249371168567344,0.1690115026410558,0.2457988990764781,0.36016345174682357,0.49284009177306154,0.5954894107682618,0.61695272953713,0.5408208496976252,0.39975533292076904,0.2520769980969108,0.05580607348342506,0.028468246988160417,0.021644557503143338,0.020870066653576227,0.02120024271582089,0.020277117418473736,0.017982875680256763,0.015638860822417743,0.014715379685997924,0.015620715493845342,0.017423563036400073,0.0186806171563383,0.018578420879847166,0.017428249318108987,0.016323463819211924,0.01647062131621042,0.018655353042321906,0.02298271504843933,0.029696056918057916,0.04191958411532769,0.0695260455273792,0.12958744429678765,0.23675052748260111,0.3839499297561328,0.5298289744477096,0.6138586617533188,0.5959727914191856,0.48749465017726606,0.3434318290493815,0.2229721376037546,0.15467424963507778,0.1340091861725716,0.14542682455447664,0.17981187054655254,0.23122679098083856,0.28419866515817027,0.3131675422479116,0.2987041349929565,0.2432587132825867,0.16923812648630607,0.10213813303843217,0.05562731975936154,0.029492121264314396,0.016867358419577427,0.011385170609967483,0.009739180240108457,0.004843561888778828,0.0059494925977014385,0.007566630088846478,0.008730187054580448,0.008933304651434638,0.008372972633725414,0.008048341451210401,0.009259365782643265,0.01265047020666372,0.017478479091580706,0.02185228119349511,0.023893068040993354,0.02289651229410969,0.019626308526009297,0.015635404258574816,0.01225601635184784,0.010186535949139038,0.010140089167973241,0.014460292220188199,0.029319716182569018,0.06675394793829909,0.1427757289952966,0.2661326643313272,0.4204448070349529,0.5576985476280167,0.6201630810148887,0.5796374234315722,0.4584885340080062,0.31161616255533797,0.18781463918820798,0.10596538380902638,0.05963264324792671,0.03473572347488386,0.02113093628893322,0.013925986737081175,0.010674771106808837,0.00960784350606935,0.00936734980376169,0.009148348750172685,0.008605969599886978,0.007670627323441305,0.006449184603787099,0.005172197468575284,0.004128558312917507,0.003558265408856413,0.0035261873102010034,0.0038720220333056466,0.004323020740424681,0.004689528253272806,0.004943634033748438,0.005110600574013385,0.005339194647909341,0.013137201766259805,0.012517425330100083,0.011958593118877201,0.011714478134591177,0.011771750540148807,0.011886005578984258,0.011665945022040918,0.0107581655790127,0.009202617655963143,0.007629925021257948,0.006913703431125713,0.0075309985039996395,0.009250413412627834,0.011432231415080405,0.013687149391007054,0.01679697678097116,0.024239683352040072,0.044456570687404155,0.09211487468145889,0.18309049773598318,0.31952406325296057,0.4735438135583146,0.5898207463004088,0.6158101681902188,0.5387315534856612,0.39540728415583276,0.24506457576923293,0.1310058450419832,0.06380367020214556,0.031571382854521726,0.018172972945989407,0.012999723654281084,0.011411283364415381,0.011887103160972155,0.013704815618177418,0.015912951385656782,0.017391329498812544,0.017413411503434537,0.016157728507099144,0.01463165420432968,0.013959936502947747,0.014576529204693777,0.01599648815717502,0.017295772448020857,0.017720813163703577,0.0169925042181572,0.01544558964090683,0.01402153182217212,0.013785846199518921,0.017480443105211175,0.03759872056339801,0.04249925363627471,0.045346231112057385,0.045864425319796294,0.04524188639159718,0.0452231005377577,0.047243292614121954,0.051890620685034396,0.058202896665399795,0.06316655593785787,0.06301704379140784,0.056318162802883286,0.04575355838657679,0.03624249270771934,0.03122497391120542,0.03106141111907505,0.035174897594445316,0.04633079221484388,0.07446187513460799,0.13601782167209253,0.24361469092162916,0.3876516323222323,0.5263595428394099,0.6026144971511518,0.5811094497536246,0.47369190044209447,0.3292027794284511,0.1989870991410542,0.10965195491939587,0.06083657725279475,0.03933212945329573,0.03201837318184273,0.030378211563151694,0.02981479137127358,0.028336600161495876,0.025735705477816173,0.017772226703460337,0.005687689571649903,0.005279881885046393,0.005363637213285462,0.006029637282898363,0.0073275100324711115,0.009722886249996255,0.013477706923164879,0.017973246574302354,0.021704482272256072,0.02318222816576307,0.022022820247893128,0.01929454331856109,0.016781958060434473,0.015724621200715872,0.016154349704622935,0.01755066718997277,0.02031654617211193,0.026711373443887347,0.04068068216684073,0.06778652548311383,0.11666098754697078,0.1987119764497247,0.3193496747588951,0.4619842337113517,0.5817240379452161,0.6253570323908053,0.5680418113944854,0.433705733932672,0.2783780596036192,0.15285627823874096,0.07890247599643097,0.05262294589939322,0.06015837517311046,0.08727892638339273,0.11993665247584948,0.14367022302083948,0.14751104269586637,0.12960659653352796,0.09830112689109469,0.06655668302367394,0.04417170233177197,0.03358470878828986,0.031058097414736297,0.03077689193792416,0.028535927503328666,0.023311999758003635,0.016672534238180756,0.010989671077555137,0.00785316726785538,0.007622832651220292,0.00998349547123925,0.014649264171927552,0.0758011768951828,0.12071730066520649,0.14614241448680648,0.15335087479296577,0.1374755545093265,0.10512586288837762,0.06973804731187211,0.042279696587034364,0.025989053178804403,0.019598261421466256,0.02189762056472684,0.06090137908875767,0.07678172833501751,0.06967437617818555,0.07592737397700365,0.07909950620666582,0.08895032217541991,0.11941711796716492,0.1819906818619731,0.2785771127234109,0.39533614424734176,0.5026570330242855,0.5662332312199911,0.5646747164675677,0.5011699399770343,0.40063353984853484,0.29471291067005756,0.20574368690780276,0.14103217690712735,0.09897231737360482,0.07786164510790496,0.07964957692262757,0.10985419963320808,0.17662380661268973,0.2849609821049271,0.42349255064717223,0.5552581177222043,0.629977015805063,0.6164342477185352,0.5277530641781706,0.41277715954197813,0.318649651938763,0.13675506770179255,0.11232774784149453,0.09376541224336676,0.07510935557958477,0.06201935805336352,0.05949607437691582,0.06666137857105515,0.0770759827811546,0.08285666471753945,0.035522370195376005,0.02716010695153496,0.020579654076779277,0.016209068986988193,0.013650825546954173,0.012611913978015135,0.012999684025079759,0.015164578901344261,0.019258815076726672,0.025301971238759848,0.03468760413016646,0.051809810544115036,0.08289791574895035,0.13036686647522758,0.1860173777939087,0.23070639186079545,0.24438660794789238,0.22009586461164685,0.17045413409019985,0.12206716041790577,0.10444853668309723,0.14096876248655368,0.24025772043961113,0.3846427861761482,0.5254417480259448,0.6012968998661969,0.5751620690723577,0.4598329487615289,0.3076258609175533,0.17407344854065326,0.08823576632050402,0.0493488577234394,0.041335870855656855,0.04682366149662582,0.053189410873585835,0.05365764723644135,0.04696929725257601,0.0359015297684422,0.024734533222234192,0.016742866856677436,0.012825919867552134,0.011720309722194306,0.011499746272715695,0.01120706980533297,0.011237915940666095,0.01234499463236505,0.014570927138550132,0.01718126403162379,0.019252361050766945,0.02013515835600683,0.01979948047755032,0.019176717937536773,0.04163363760956209,0.2096842934546305,0.1420645712531294,0.13264164296345063,0.132673822591748,0.12539494684242924,0.1128371512002164,0.10063278344460287,0.09144885923516823,0.0851632878347743,0.08204870828704158,0.04817828853972557,0.04006154458755149,0.04488541129690742,0.0477951901840857,0.04623612262415552,0.040680216768996566,0.03489341169210034,0.03357509943360962,0.03968776035820266,0.05363210654726906,0.07425512143215897,0.10092084482521375,0.13513400224612115,0.17863914012955184,0.2271934242513608,0.2663239957104824,0.27704962272636396,0.24983478498103665,0.1946216001188007,0.13744900445548436,0.10834991365287386,0.13160763268785225,0.219032222363505,0.35932117170065075,0.509377540564976,0.6064824000460515,0.6040536559971886,0.5035705259317067,0.35252620418660313,0.20988819672408746,0.11070663599969374,0.057279010507168014,0.033995123786127374,0.024721996977644243,0.020127254378118842,0.016894855814695006,0.014755158193535223,0.014205618185359276,0.015505121541746034,0.018737247368189103,0.02402213721180039,0.031055167764035028,0.038359922165595355,0.04364717397121728,0.045421822681787,0.044023991996975986,0.04086398226180071,0.036979009187574484,0.03264149039298894,0.028160585708683745,0.02473791547554381,0.024645787492334398,0.039368685300516224,0.10688620616039085,0.09626372559211208,0.057011609986306944,0.055395916751583375,0.049017056895980494,0.043404100794211635,0.0472773896119798,0.06978493046683329,0.11472352641040953,0.17405702538248896,0.22729537674869976,0.25160609780227366,0.2374789332594616,0.1965374316144988,0.15341340203577578,0.12826379863042198,0.12468834796163676,0.1314902536760797,0.1343989577200454,0.1281958687324892,0.12226451855554567,0.13654294400489833,0.18947443920900695,0.28456281148922113,0.40362699418875897,0.510575669875388,0.5654295622605552,0.5441221559792285,0.45289834515649774,0.3254560591999432,0.2032892779684819,0.11385026572693674,0.06219547978232719,0.03873161241535068,0.032261551473615986,0.03690410375812367,0.05063080036745865,0.0704829031098134,0.09029729457744376,0.10271280201531277,0.10318488690316957,0.09222687989141722,0.07475206577808956,0.05785932869122972,0.04784555809387798,0.04706259333747742,0.05297020694806463,0.060874974887845934,0.06752782515558485,0.07166289020632441,0.07221354589203198,0.0680321296530083,0.09137413668307473,0.07855788352627353,0.12356203451842898,0.13940743458851065,0.18213991639257035,0.2497913905077779,0.34115186993008434,0.45336666273133763,0.32275105256727804,0.1207412277131483,0.11267053045609493,0.0991312290281882,0.07557207629472125,0.050802875376083856,0.03181276935904768,0.02081279252687493,0.01646607454267802,0.016314351896116217,0.017934668075352878,0.01922204755723338,0.01881055853177496,0.016501186992760132,0.013206477439078415,0.010404855147225368,0.00931325969386198,0.010260400888237512,0.013063071909201824,0.019030562059201377,0.03411532833417862,0.07127686804358807,0.14751004394577666,0.27110979493208903,0.4238701699156205,0.5563224557137809,0.6108815391105427,0.5606934856591318,0.43004171735178426,0.27635864631546464,0.15149871057105216,0.07709007319081197,0.04768804469515306,0.0467409660563569,0.05754622278365272,0.06693191575346774,0.0673047857604897,0.05827753993747344,0.04514561210356412,0.03439665797834275,0.02985256281793837,0.03197848580268128,0.03914910629082342,0.04843175156638216,0.05580521952955578,0.05742737064276997,0.05188940708708072,0.04125504670077837,0.029603400415128166,0.020462719360237483,0.015390793185612216,0.014268174709751448,0.01690065764039345,0.060421455512508784,0.07491656395069711,0.07954912204234064,0.07752602751370674,0.07615449140169603,0.08915795153443952,0.13184681217296182,0.21534730527995294,0.33922385667096866,0.48287897549812314,0.605749410580086,0.3119396226622529,0.2575435049143982,0.23262691523971088,0.2021799089294369,0.17352703782823597,0.14505465983388582,0.11497679916980381,0.08779453217368564,0.08025445216806666,0.09921396387952189,0.14078059334660006,0.18658856261975654,0.2133458446659433,0.2073072605792402,0.17226452895172426,0.12481422699847165,0.08234895468273715,0.05455703577476276,0.043660269519719366,0.05025512517230666,0.0794675089431197,0.14281814901929285,0.25075075238460515,0.3964655852366264,0.5457743761355616,0.6520550991157781,0.6917097104611785,0.6859836494108117,0.6816016760775947,0.7058655337658781,0.7417446805560509,0.7466222131671663,0.5494371149806243,0.4617275466810364,0.353561684239796,0.24959455386018128,0.16471468209452692,0.10362493441872278,0.06513157156955571,0.04606512696281927,0.043082805774478095,0.05221652845183303,0.06736944070542458,0.08031500851183222,0.08373132818265668,0.07543115721446762,0.0597038622347942,0.044017632061850644,0.034036436863744135,0.0318413210554345,0.03917292368292177,0.06304948214469774,0.11793854243552367,0.21637412141662174,0.23414642905821917,0.32977195822696975,0.3906222526201502,0.3677411808505749,0.30364505365279054,0.2086208625147341,0.12016092345341553,0.060848005585596075,0.03207659498694129,0.02317813652079253,0.02271242811729031,0.023542345412361526,0.022672590388234874,0.01976633566832763,0.016069468945406472,0.013561085328910868,0.014407113469254674,0.021533414906902443,0.04068839368662154,0.08190498990394042,0.15689330364674323,0.2711345853594403,0.41562746539005907,0.567064822540438,0.698648846175814,0.7915393577367263,0.8352789971243941,0.8206053254970005,0.7397803436974698,0.5989008230832076,0.4268618379188501,0.26528525215441895,0.1457983302709797,0.07608941417806973,0.044507821357993145,0.033485823037880835,0.02959905228406158,0.027150525505473262,0.027773026019530644,0.03790884669663663,0.06415631665971275,0.1069415732757906,0.15584830298069743,0.1921519203653061,0.1995290844146846,0.17549201915556897,0.13374183174591178,0.09491131258332393,0.07312739278617557,0.0687940996126653,0.07178396470997049,0.07075352565646348,0.060994714352430424,0.050101604269866894,0.037287868610243614,0.08849456181174374,0.09654182393906517,0.11583251042773007,0.12900471728370722,0.12631371576228248,0.10974632980349176,0.08798618254270385,0.06857430309552638,0.05482747133773031,0.047849583309542684,0.04820872912095124,0.0545260017004374,0.06208761425457605,0.06526738970446766,0.0636378524519758,0.06442069659092135,0.0888033506804178,0.10278737638623966,0.12433664266034272,0.14143010729265817,0.14401495380463852,0.13198513686220764,0.11880308798000076,0.128678074425434,0.1855339232577488,0.29568787640769834,0.4348829945308745,0.5539535501020356,0.6046870046761343,0.5680510883281463,0.4630832484859211,0.33157954566907477,0.21312971968859754,0.12867041161783643,0.07981111653273208,0.058417623282925324,0.05640492170227881,0.0693561112775079,0.0946664418821422,0.12827758896685054,0.16313991111770568,0.19077493199167944,0.20665956791674234,0.21755279013587095,0.24329176836174476,0.30483220115697796,0.4020616511922749,0.5017202960164258,0.5535549267984047,0.5254321243267205,0.4263649266385075,0.29761340565981,0.1831417407098255,0.10638910590240816,0.07643121565462978,0.061170280317294434,0.06057356284724037,0.06815976842794243,0.08466738631028674,0.11099484420104598,0.14187883971473522,0.16743631285325525,0.18071325080338568,0.18281248015031326,0.18075654554348214,65.47661150624442};
//    for (int i = 0; i < temp2.size(); ++i) {
//        y[i] = temp2[i];
//    }
//
//    vector<double> temp3 = {3020.6391, 3024.0325, 3037.3887, 3047.6043, 3057.4456, 3059.0856, 3225.785, 3243.6887, 3350.9243, 3376.4359, 3388.5309, 3399.3335, 3404.3535, 3407.4585, 3413.1312, 3427.1192, 3443.8762, 3465.8603, 3475.45, 3476.7016, 3480.5055, 3490.5737, 3509.7785, 3519.9936, 3535.3196, 3541.0832, 3548.5144, 3559.5081, 3561.0304, 3565.3786, 3576.6156, 3581.1925, 3588.4407, 3608.8587, 3618.7676, 3631.4629, 3647.8424, 3679.9132, 3687.4564, 3705.5657, 3709.2459, 3718.2065, 3719.9346, 3722.5625, 3727.6187, 3729.3087, 3733.3169, 3734.8636, 3737.1313, 3745.5608, 3748.2617, 3749.4847, 3758.2324, 3763.7885, 3767.1914, 3812.9641, 3815.8397, 3820.4251, 3824.4436, 3825.8805, 3827.8226, 3834.2222, 3856.3717, 3859.9114, 3868.5284, 3878.573, 3886.282, 3895.6558, 3899.7073, 3902.9452, 3906.4794, 3920.2577, 3922.9115, 3925.7188, 3930.2962, 3946.0971, 3948.9789, 3969.257, 3979.3559, 3994.7918, 4005.2414, 4033.8093, 4042.8937, 4044.4179, 4045.813, 4052.9208, 4063.5939, 4103.9121, 4118.5442, 4131.7235, 4143.8688, 4158.5905, 4164.1795, 4198.3036, 4200.6745, 4216.1828, 4222.6373, 4237.2198, 4259.3619, 4271.7593, 4277.5282, 4294.1243, 4300.1008, 4307.9015, 4315.0837, 4325.7615, 4331.1995, 4333.5612, 4335.3379, 4337.0708, 4345.168, 4348.064, 4352.2049, 4362.0662, 4367.8316, 4375.9294, 4379.6668, 4383.5445, 4385.0566, 4400.9863, 4404.7499, 4415.1222, 4426.0011, 4427.3039, 4430.189, 4433.838, 4439.4614, 4448.8792, 4461.6521, 4466.5508, 4474.7594, 4481.8107, 4489.7389, 4490.9816, 4498.5384, 4502.9268, 4510.7332, 4522.323, 4528.6133, 4530.5523, 4545.0519, 4579.3495, 4589.8978, 4596.0967, 4598.7627, 4609.5673, 4628.4409, 4637.2328, 4647.4329, 4657.9012, 4702.3161, 4726.8683, 4732.0532, 4735.9058, 4764.8646, 4806.0205, 4847.8095, 4859.7406, 4879.8635, 4889.0422, 4891.4919, 4920.5018, 4933.2091, 4957.5966, 4965.0795, 4972.1597, 5006.1175, 5009.3344, 5012.0674, 5017.1628, 5051.6336, 5062.0371, 5083.3377, 5090.4951, 5110.3849, 5125.7654, 5141.7827, 5145.3083, 5162.2846, 5167.4873, 5171.5953, 5187.7462, 5194.9412, 5221.271, 5227.1697, 5232.9394, 5266.5546, 5269.5366, 5283.6206, 5302.2989, 5324.1782, 5328.0376, 5341.0233, 5371.4892, 5397.1269, 5405.7741, 5415.1997, 5421.3517, 5424.0686, 5429.6955, 5434.5228, 5439.9891, 5446.8937, 5451.652, 5455.609, 5495.8738, 5514.376, 5524.957, 5558.702, 5569.6177, 5577.6845, 5586.7553, 5606.733, 5615.6436, 5648.6863, 5650.7043, 5681.9001, 5691.6612, 5739.5196, 5783.536, 5786.5553, 5802.0798, 5812.7592, 5834.2633, 5860.3103, 5882.6242, 5888.5841, 5912.0853, 5916.5992, 5927.1258, 5928.813, 5942.6686, 5949.2583, 5964.4723, 5968.3199, 5971.6008, 5987.3016, 5998.9987, 6005.7242, 6013.6777, 6025.15, 6032.1274, 6043.2233, 6046.8977, 6052.7229, 6059.3725, 6081.2433, 6085.8797, 6090.7848, 6098.8031, 6105.6351, 6114.9234, 6123.3619, 6127.416, 6145.4411, 6155.2385, 6165.1232, 6170.174, 6172.2778, 6191.5583, 6201.1002, 6212.5031, 6215.9383, 6230.726, 6243.1201, 6246.3172, 6248.4055, 6252.5537, 6296.8722, 6307.657, 6324.4163, 6364.8937, 6369.5748, 6384.7169, 6399.9995, 6403.0128, 6411.6468, 6416.3071, 6437.6003, 6466.5526, 6483.0825, 6538.112, 6604.8534, 6614.3475, 6620.9665, 6632.0837, 6638.2207, 6639.7403, 6643.6976, 6656.9386, 6660.6761, 6664.051, 6666.3588, 6677.2817, 6684.2929, 6719.2184, 6752.8335, 6766.6117, 6861.2688, 6871.2891, 6879.5824, 6888.1742, 6937.6642, 6951.4776, 6965.4307, 7030.2514, 7067.2181, 7107.4778, 7125.82, 7147.0416, 7158.8387, 7206.9804, 7265.1724, 7272.9359, 7311.7159, 7316.005, 7353.293, 7372.1184, 7383.9805, 7392.9801, 7412.3368, 7425.2942, 7435.3683, 7471.1641, 7484.3267, 7503.8691, 7514.6518, 7589.3151, 7635.106, 7670.0575, 7798.5604, 7868.1946, 7891.075, 7916.442, 7948.1764, 7948.1964, 8006.1567, 8014.7857, 8037.2183, 8046.1169, 8053.3085, 8066.6046, 8103.6931, 8115.311, 8143.505, 8203.4352, 8264.5225, 8327.0526, 8384.724, 8387.77, 8408.2096, 8424.6475, 8490.3065, 8521.4422, 8605.7762, 8620.4602, 8667.9442, 8688.6213, 8761.6862, 8799.0875, 8962.1468, 9008.4636, 9017.5912, 9122.9674, 9194.6385, 9224.4992, 9291.5313, 9354.2198, 9508.4513, 9657.7863, 9657.7863, 9784.5028, 10470.0535};
//    for (int i = 0; i < temp3.size(); ++i) {
//        l[i] = temp3[i];
//    }
//
//
////    double dsum = debug_sum(x, y, l, int(lines_size), int(compspec_size), -4.739e-09, 6.181e-06,  1705.0458, 3591.26+1705.0458/2);
////    double dsum = debug_sum(x, y, l, int(lines_size), int(compspec_size), -8.5e-11, -5.8e-06, 1698.3, 3635.75+1698.3/2);
//    double dsum = debug_sum(x, y, l, int(lines_size), int(compspec_size), -5e-12, -8e-06, 1728.9, 3588.45+1728.9/2);
//
//    cout << "Debug sum: " << setprecision(numeric_limits<double>::digits10 + 1) << dsum << endl;
//
//    auto[a,b,c,d] = fitlines(x, y, l, int(lines_size), int(compspec_size), center, extent, quadratic_ext, cubic_ext, 100, 50, 100, 100, 100., 0.05, 2.e-5, 2.5e-10, 25, 3);
//
//    cout << setprecision(numeric_limits<double>::digits10 + 1) << a << " " << b << " " << c << " " << d << endl;
//    cout << setprecision(numeric_limits<double>::digits10 + 1) << cubic_ext+a << ", " << quadratic_ext+b  << ", " << center-(extent*(1+d))/2+c << ", " << extent*(1+d) << endl;
//    cout << endl;
//
//    // Get the ending time point
//    auto end = chrono::high_resolution_clock::now();
//
//    // Calculate the duration in milliseconds
//    chrono::duration<double, milli> duration = end - start;
//
//    cout << "Execution time: " << duration.count()/1000 << " s" << endl;
//    return 0;
//}

void readfile(const string& filename, double*& array, size_t& length) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Could not open the file " << filename << endl;
        array = nullptr;
        length = 0;
        return;
    }

    vector<double> values;
    string line;
    while (getline(infile, line)) {
        istringstream iss(line);
        double value;
        if (iss >> value) {
            values.push_back(value);
        } else {
            cerr << "Invalid line in file: "<< filename << " : " << line << endl;
        }
    }

    length = values.size();
    array = new double[length];
    for (size_t i = 0; i < length; ++i) {
        array[i] = values[i];
    }

    infile.close();
}

void writeOutput(const string& filename, double a, double b, double c, double d) {
    // Create an ofstream object for file output
    ofstream outFile(filename);

    // Check if the file was successfully opened
    if (!outFile) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    // Write the doubles to the file, one per line
    outFile << setprecision(numeric_limits<double>::digits10 + 1) << a << endl;
    outFile << setprecision(numeric_limits<double>::digits10 + 1) << b << endl;
    outFile << setprecision(numeric_limits<double>::digits10 + 1) << c << endl;
    outFile << setprecision(numeric_limits<double>::digits10 + 1) << d << endl;

    // Close the file
    outFile.close();

    // Optional: check if the file was successfully closed
    if (!outFile) {
        cerr << "Error closing file: " << filename << endl;
    }
}

int main(int argc, char *argv[]) {
    auto start = chrono::high_resolution_clock::now();
    if (argc != 6) {
        cerr << "ERROR: Invalid number of arguments! Please pass filenames for x,y arrays, lines and arguments.";
        return -1;
    }

    double *compspec_x = nullptr;
    double *compspec_y = nullptr;
    double *lines = nullptr;
    double *arguments = nullptr;
    size_t compspec_length = 0;
    size_t lines_length = 0;
    size_t arguments_length = 0;

    if (stoi(argv[5]) == 0) {
        readfile(argv[1], compspec_x, compspec_length);
        readfile(argv[2], compspec_y, compspec_length);
        readfile(argv[3], lines, lines_length);
        readfile(argv[4], arguments, arguments_length);

        double center = arguments[0];
        double extent = arguments[1];
        double quadratic_ext = arguments[2];
        double cubic_ext = arguments[3];

        auto [a, b, c, d] = fitlines(compspec_x, compspec_y, lines, static_cast<int>(lines_length),
                                     static_cast<int>(compspec_length),
                                     arguments[0], arguments[1], arguments[2], arguments[3],
                                     static_cast<size_t>(arguments[4]),
                                     static_cast<size_t>(arguments[5]), static_cast<size_t>(arguments[6]),
                                     static_cast<size_t>(arguments[7]),
                                     arguments[8], arguments[9], arguments[10], arguments[11], arguments[12],
                                     static_cast<int>(arguments[13]));

        a = cubic_ext + a;
        b = quadratic_ext + b;
        c = center - (extent * (1 + d)) / 2 + c;
        d = extent * (1 + d);
        writeOutput("temp/output.txt", a, b, d, c);

        // Get the ending time point
        auto end = chrono::high_resolution_clock::now();

        // Calculate the duration in milliseconds
        chrono::duration<double, milli> duration = end - start;

        cout << "Found wavelength solution in: " << duration.count() / 1000 << " s" << endl;
        return 0;
    }
    else{
        cout << "Using markov chain algorithm..." << endl;

        readfile(argv[1], compspec_x, compspec_length);
        readfile(argv[2], compspec_y, compspec_length);
        readfile(argv[3], lines, lines_length);
        readfile(argv[4], arguments, arguments_length);

        #pragma omp parallel for
        for (int i = 0; i < omp_get_num_procs(); ++i) {
            // Create a unique filename for each iteration
            string output_file = "mcmkc_output" + to_string(i) + ".txt";
            fitlines_mkcmk(compspec_x, compspec_y, lines, static_cast<int>(arguments[0]),
                           static_cast<int>(arguments[1]), static_cast<int>(arguments[2]),
                           arguments[3], arguments[4], arguments[5], arguments[6], arguments[7],
                           arguments[8], arguments[9], arguments[10], arguments[11], arguments[12], arguments[13],
                           arguments[14], output_file);
        }

        // Get the ending time point
        auto end = chrono::high_resolution_clock::now();

        // Calculate the duration in milliseconds
        chrono::duration<double, milli> duration = end - start;

        cout << "Found wavelength solution in: " << duration.count() / 1000 << " s" << endl;
        return 0;
    }
}
