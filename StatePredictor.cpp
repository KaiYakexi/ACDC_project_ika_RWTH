#include "StatePredictor.h"
#include "definitions/utility/ika_utilities.h"
#include "math.h"
#include "stdio.h"
using namespace std;
StatePredictor::StatePredictor(std::shared_ptr<Data> data, std::string name)
: AbstractFusionModule(data, name) {}
void StatePredictor::runSingleSensor()

{

// TODO: INSERT CODE HERE
Eigen::MatrixXf F_delta = Eigen::MatrixXf::Zero(8,8);
Eigen::MatrixXf F = Eigen::MatrixXf::Zero(8,8);
Eigen::MatrixXf F_global = Eigen::MatrixXf::Identity(10,10);
cout << F_global;
Eigen::MatrixXf Q = data_->Q_timevar_ * data_->prediction_gap_in_seconds;
Eigen::VectorXf x_hat_G_8;
x_hat_G_8 << 0,0,0,0,0,0,0,0;
Eigen::VectorXf x_hat_G_10;
x_hat_G_10 << 0,0,0,0,0,0,0,0,0,0;

for (auto &globalObject : data_->object_list_fused.objects) {
auto x_hat_G = IkaUtilities::getEigenStateVec(&globalObject); // estimated global state
auto v_x = x_hat_G[3];
auto v_y = x_hat_G[4];

// auto theta_yaw = globalObject->fMean[(int)definitions::ctra_model::heading];
// auto omega_yaw = IkaUtilities::getObjectYawrate(globalObject);

auto theta_yaw = 0;
auto omega_yaw = 1;
auto v = sqrt(pow(v_x, 2) + pow(v_y, 2));
x_hat_G_8[0] = x_hat_G[0];
x_hat_G_8[1] = x_hat_G[1];
x_hat_G_8[2] = v;
//x_hat_G_8[3] = theta_yaw;
//x_hat_G_8[4] = omega_yaw;
x_hat_G_8[3] = theta_yaw;
x_hat_G_8[4] = omega_yaw;
x_hat_G_8[5] = x_hat_G[7];
x_hat_G_8[6] = x_hat_G[8];
x_hat_G_8[7] = x_hat_G[9];
cout << x_hat_G_8;
auto alpha_1 = (sin(theta_yaw + omega_yaw * data_->prediction_gap_in_seconds) - sin(theta_yaw)) / omega_yaw;
auto alpha_2 = (cos(theta_yaw) - cos(theta_yaw + omega_yaw * data_->prediction_gap_in_seconds)) / omega_yaw;
auto r_1 = v * (cos(theta_yaw + omega_yaw * data_->prediction_gap_in_seconds) - cos(theta_yaw)) / omega_yaw;
auto r_2 = -1 * v * (sin(theta_yaw + omega_yaw * data_->prediction_gap_in_seconds) - sin(theta_yaw)) / omega_yaw;
auto b_1 = -1 * v * (sin(theta_yaw + omega_yaw * data_->prediction_gap_in_seconds) - sin(theta_yaw)) / pow(omega_yaw, 2);
auto b_2 = -1 * v * (cos(theta_yaw) - cos(theta_yaw + omega_yaw * data_->prediction_gap_in_seconds)) / pow(omega_yaw, 2);
F_delta(0,2) = alpha_1;
F_delta(1,2) = alpha_2;
F_delta(0,3) = r_1;
F_delta(1,3) = r_2;
F_delta(0,4) = b_1;
F_delta(1,4) = b_2;
F_delta(3,4) = data_->prediction_gap_in_seconds;
F = data_->F_const_ + F_delta; // F is chao's Jacobian Matrix
x_hat_G_8 = F * x_hat_G_8;
x_hat_G_10[0] = x_hat_G_8[0];
x_hat_G_10[1] = x_hat_G_8[1];
x_hat_G_10[2] = 0;
x_hat_G_10[3] = x_hat_G_8[2]*cos(theta_yaw + omega_yaw * data_->prediction_gap_in_seconds);
x_hat_G_10[4] = x_hat_G_8[2]*sin(theta_yaw + omega_yaw * data_->prediction_gap_in_seconds);
x_hat_G_10[5] = 0;
x_hat_G_10[6] = 0;
x_hat_G_10[7] = x_hat_G_8[5];
x_hat_G_10[8] = x_hat_G_8[6];
x_hat_G_10[9] = x_hat_G_8[7];

x_hat_G = x_hat_G_10.transpose(); // value of x is edited in place of ikaObject memory
globalObject.P() = F_global * globalObject.P() * F_global.transpose() + Q;

// update global object time stamp
globalObject.header.stamp = data_->object_list_measured.header.stamp;
}

// update global object list time stamp
data_->object_list_fused.header.stamp = data_->object_list_measured.header.stamp;
}
