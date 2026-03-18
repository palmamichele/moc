#include "fmca/FMCA/Clustering"
#include "fmca/FMCA/src/ModulusOfContinuity/ExactDiscreteModulusOfContinuity.h"
#include "fmca/FMCA/src/util/Tictoc.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

FMCA::Matrix csvToEigen(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  std::vector<std::vector<double>> rows;
  std::string line;

  // Read CSV line by line
  while (std::getline(file, line)) {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(cell)); // parse string to double
    }
    rows.push_back(row);
  }

  if (rows.empty())
    return FMCA::Matrix(); // empty file

  size_t numRows = rows.size();
  size_t numCols = rows[0].size();

  FMCA::Matrix mat(numCols, numRows); // each row becomes a column
  for (size_t i = 0; i < numRows; ++i) {
    for (size_t j = 0; j < numCols; ++j) {
      mat(j, i) = rows[i][j];
    }
  }

  return mat;
}

int main() {
  FMCA::Tictoc T;
  std::string path = "../data/mnist-100-3";
  FMCA::Matrix X = csvToEigen(path + "/X.csv");

  FMCA::Matrix F = csvToEigen(path + "/F_0.csv");

  FMCA::ExactDiscreteModulusOfContinuity dmoc;
  FMCA::Scalar delta_step = 1;

  FMCA::Scalar r = 0.0005;
  FMCA::Scalar R = 2;
  FMCA::Scalar min_csize = 1;

  T.tic();
  dmoc.init(X, F, 0, "EUCLIDEAN", "EUCLIDEAN", "TRICK");
  dmoc.computeMocPlot(X, F, delta_step);
  std::vector<double> omega_vec = dmoc.getOmegaT();
  T.toc("eval: ");
  FMCA::Vector omega_eigen =
      Eigen::Map<FMCA::Vector>(omega_vec.data(), omega_vec.size());

  std::cout << "omega_t: " << omega_eigen;

  T.tic();
  dmoc.init(X, F, 0, "EUCLIDEAN", "EUCLIDEAN", "NO");
  dmoc.computeMocPlot(X, F, delta_step);
  omega_vec = dmoc.getOmegaT();
  T.toc("eval with no trick: ");

  // T.tic();
  // dmoc.init(X, F, omega_vec.size(), "EUCLIDEAN", "EUCLIDEAN", "NO");
  // dmoc.computeMocPlot(X, F, delta_step);
  // omega_vec = dmoc.getOmegaT();
  // T.toc("eval knowing  the max distance: ");

  // T.tic();
  // emoc.init<FMCA::ClusterTree>(X, F, r, R, omega_vec.size(), min_csize,
  //                              "EUCLIDEAN", "EUCLIDEAN", true);

  // for (FMCA::Scalar t = 0; t <= omega_vec.size(); t += delta_step) {
  //   FMCA::Scalar omega_val = dmoc.template omega<FMCA::ClusterTree>(t, X, F);
  //   std::cout << t << " w" << omega_val << std::endl;
  // }

  // T.toc("eval epsilon knowing the max distance: ");
}