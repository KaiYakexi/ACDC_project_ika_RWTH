#include "StateFuser.h"
#include "data/Params.h"
#include <definitions/utility/ika_utilities.h>
#include "math.h"

StateFuser::StateFuser(std::shared_ptr<Data> data, std::string name)
: AbstractFusionModule(data, name) {}

Eigen::MatrixXf StateFuser::constructMeasurementMatrix() {

  // find valid object in measured object list
  Eigen::VectorXf variance;
  bool foundValidObject = false;
  for (auto &measuredObject: data_->object_list_measured.objects) {
    if (measuredObject.bObjectValid) {
      variance = IkaUtilities::getEigenVarianceVec(&measuredObject);
      foundValidObject = true;
    }
  }

  if (!foundValidObject) {
    return Eigen::MatrixXf::Zero(0,variance.size());
  }

  int row_size = 0;
  for (int i = 0; i < variance.size(); ++i) {
    if (variance[i] >= 0) {
      ++row_size;
    }
  }
  if (row_size == 0) return Eigen::MatrixXf::Zero(0, variance.size());

  Eigen::MatrixXf C = Eigen::MatrixXf::Zero(long(row_size), variance.size());

  int row = 0;
  for (int col = 0; col < variance.size() && row < row_size; ++col) {
    if (variance[col] >= 0) {
      C(row, col) = 1;
      row++;
    }
  }
  return C;
}

void StateFuser::runSingleSensor() {

  Eigen::MatrixXf C = constructMeasurementMatrix();
  //nonlinear measurement h_x_G_

   for (auto globalObject : data_->object_list_measured.objects){
    auto vector_v = IkaUtilities::getObjectVelocity(globalObject);
    auto vector_xyz = IkaUtilities::getObjectPosition(globalObject);
    float x = vector_xyz[0];
    float y = vector_xyz[1];
    float v_x = vector_v[0];
    float v_y = vector_v[1];
    break;
  }

  auto h_x_G_ = Eigen::MatrixXf(3,1);
  h_x_G_(0,0) = sqrt(pow(x,2) + pow(y, 2));
  h_x_G_(1,0) = std::atan(y/x);
  h_x_G_(2,0) = (x * v_x + y * v_y) / sqrt(pow(x, 2) + pow(y, 2));
  
 //Jacobian_H_
 auto v = sqrt(pow(v_x, 2) + pow(v_y, 2));
 auto Jacobian_H_ = Eigen::MatrixXf(3, 5);
 Jacobian_H_(0, 0) = x / sqrt(pow(x,2) + pow(y, 2));
 Jacobian_H_(0, 1) = y / sqrt(pow(x,2) + pow(y, 2));
 Jacobian_H_(0, 2) = 0;
 Jacobian_H_(0, 3) = 0;
 Jacobian_H_(0, 4) = 0;
 Jacobian_H_(0, 5) = 0;
 Jacobian_H_(0, 6) = 0;
 Jacobian_H_(0, 7) = 0;
 Jacobian_H_(0, 8) = 0;
 Jacobian_H_(0, 9) = 0;
 Jacobian_H_(1, 0) = (1 - y) / sqrt(pow(x,2) + pow(y, 2));
 Jacobian_H_(1, 1) = 1 / (x + pow(y, 2) / x);
 Jacobian_H_(1, 2) = 0;
 Jacobian_H_(1, 3) = 0;
 Jacobian_H_(1, 4) = 0;
 Jacobian_H_(1, 5) = 0;
 Jacobian_H_(1, 6) = 0;
 Jacobian_H_(1, 7) = 0;
 Jacobian_H_(1, 8) = 0;
 Jacobian_H_(1, 9) = 0;
 Jacobian_H_(2, 0) = v_x * pow(y, 2) / (pow(x, 2) + pow(y, 2)) * sqrt(pow(x,2) + pow(y,2));
 Jacobian_H_(2, 1) = v_y * pow(x, 2) / (pow(x, 2) + pow(y, 2)) * sqrt(pow(x,2) + pow(y,2));
 Jacobian_H_(2, 2) = 0;
 Jacobian_H_(2, 3) =  (x * v / sqrt(pow(v, 2) - pow(v_y, 2)) + y * v / sqrt(pow(v, 2) - pow(v_x, 2))) / sqrt(pow(x, 2) + pow(y, 2));
 Jacobian_H_(2, 4) =  (x * v / sqrt(pow(v, 2) - pow(v_y, 2)) + y * v / sqrt(pow(v, 2) - pow(v_x, 2))) / sqrt(pow(x, 2) + pow(y, 2));
 Jacobian_H_(2, 3) = 0;
 Jacobian_H_(2, 4) = 0;
 Jacobian_H_(2, 5) = 0;
 Jacobian_H_(2, 6) = 0;
 Jacobian_H_(2, 7) = 0;
 Jacobian_H_(2, 8) = 0;
 Jacobian_H_(2, 9) = 0;
 
  int count = -1;
  for (auto &globalObject : data_->object_list_fused.objects) {
    count++;

    auto x_hat_G = IkaUtilities::getEigenStateVec(&globalObject); // predicted global state

    int measurementIndex = data_->associated_measured[count];

    if (measurementIndex < 0) {
      continue; // no associated measurement
    }

    definitions::IkaObject& measuredObject = data_->object_list_measured.objects[measurementIndex];
    auto P_S_diag = IkaUtilities::getEigenVarianceVec(&measuredObject); // predicted measured state variance

    auto R_diag = C * P_S_diag;
    Eigen::MatrixXf R = R_diag.asDiagonal();
    Eigen::MatrixXf C_transposed = C.transpose();
    Eigen::MatrixXf Jacobian_H_transposed = Jacobian_H_.transpose();
    Eigen::VectorXf x_hat_S = IkaUtilities::getEigenStateVec(&measuredObject);
    Eigen::VectorXf z = C * x_hat_S;
    Eigen::MatrixXf S = (Jacobian_H_ * globalObject.P() * Jacobian_H_transposed + R);
    Eigen::MatrixXf K = globalObject.P() * Jacobian_H_transposed * S.inverse();
    x_hat_G = x_hat_G + K * (z - (C * h_x_G_)); // value of x is edited in place of ikaObject memory

    // Update global matrix P.
    Eigen::MatrixXf K_times_H = K * Jacobian_H_;
    Eigen::MatrixXf Identity = Eigen::MatrixXf::Identity(K_times_H.rows(), K_times_H.cols());
    globalObject.P() = (Identity - K_times_H) * globalObject.P();
  }
