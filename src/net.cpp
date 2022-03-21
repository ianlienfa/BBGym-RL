// #include <torch/torch.h>
// #include <iostream>
// #include <iomanip>
// #include "third_party/matplotlibcpp/include/matplotlibcpp.h"
// namespace plt = matplotlibcpp;
// using namespace std;
// using namespace torch;

// /*
//     A DCGan implementation 
//     Discriminator: gives an probability of the image being real
//     Generator: generates an image from noise
// */

// /* The Generator Module */
// struct DCGANGeneratorImpl: nn::Module {
//     // Nets
//     nn::ConvTranspose2d conv1, conv2, conv3, conv4; // (in_channels, out_channels, kernel_size, stride, padding)
//     nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;

//     // Constructor
//     DCGANGeneratorImpl(int kNoiseSize):
//         conv1(
//             nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)
//         )
//         ,batch_norm1(256),
//         conv2(
//             nn::ConvTranspose2dOptions(256, 128, 3)
//                 .stride(2)
//                 .padding(1)
//                 .bias(false)
//         ),
//         batch_norm2(128),
//         conv3(
//             nn::ConvTranspose2dOptions(128, 64, 4)
//                 .stride(2)
//                 .padding(1)
//                 .bias(false)
//         ),
//         batch_norm3(64),
//         conv4(
//             nn::ConvTranspose2dOptions(64, 3, 4)
//                 .stride(2)
//                 .padding(1)
//                 .bias(false)                        
//         )
//     {
//         // For parameters() method
//         register_module("conv1", conv1);
//         register_module("conv2", conv2);
//         register_module("conv3", conv3);
//         register_module("conv4", conv4);
//         register_module("batch_norm1", batch_norm1);
//         register_module("batch_norm2", batch_norm2);
//         register_module("batch_norm3", batch_norm3);
//     }

//     // Forward
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(batch_norm1(conv1(x)));
//         x = torch::relu(batch_norm2(conv2(x)));
//         x = torch::relu(batch_norm3(conv3(x)));
//         x = torch::tanh(conv4(x));
//         return x;
//     }
// };

// // Wrap DCGANGeneratorImpl with shared_ptr (to call by reference)
// TORCH_MODULE(DCGANGenerator);

// nn::Sequential Discriminator(
//     nn::Conv2d(
//         // input channel: 1, output channel: 64, kernel size: 4 * 4
//         nn::Conv2dOptions(1, 64, 4)
//             .stride(2)
//             .padding(1)
//             .bias(false)
//     ),
//     // takes in slope 
//     nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)), 
//     nn::Conv2d(
//         nn::Conv2dOptions(64, 128, 4)
//             .stride(2)
//             .padding(1)
//             .bias(false)
//     ),
//     nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
//     nn::Conv2d(
//         nn::Conv2dOptions(128, 256, 4)
//             .stride(2)
//             .padding(1)
//             .bias(false)
//     ),
//     nn::BatchNorm2d(256),
//     nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
//     nn::Conv2d(
//         nn::Conv2dOptions(256, 1, 3)
//             .stride(2)
//             .padding(1)
//             .bias(false)
//     ),
//     nn::Sigmoid()   // output binary probability
// );

// template <typename T> std::string type_name();


// int main() {
//     auto dataset = torch::data::datasets::MNIST("../mnist")
//     .map(torch::data::transforms::Normalize<>(0.5, 0.5))
//     .map(torch::data::transforms::Stack<>());

//     // use std::move to avoid copying
//     auto data_loader = torch::data::make_data_loader(
//         std::move(dataset),
//         torch::data::DataLoaderOptions().batch_size(64).workers(4)
//     );

//     for(torch::data::Example<>& batch: *data_loader)
//     {
//         // A batch is actally a vector of Example objects
//         // batch.data is the tensor of the batch, size(0) gives the size of dim 0
//         std::cout << "Batch size: " << batch.data.size(0)  << " | Labels: ";
//         for(int64_t i = 0; i < batch.data.size(0); i++)
//         {
//             std::cout << batch.target[i].item<int64_t>() << " ";
//         }
//         std::cout << std::endl;
//     }
// }

#include <torch/torch.h>
#include "search_modules/Net/DDPR/NetDDPR.h"
#include "util/TorchUtil.h"

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

float max_y = NULL, min_y = NULL;
vector<float>& normalize_vec(vector<float> &vec)
{    
    vector<float>::const_iterator min_iter, max_iter;
    tie(min_iter, max_iter) = minmax_element(vec.begin(), vec.end());
    float denom = (*max_iter - *min_iter) ? (*max_iter - *min_iter) : 1;
    cout << "denom: " << denom << endl;
    for (auto &i : vec)
    {
        i = (i - *min_iter) / denom;
    }
    return vec;
} 

float denormalize(float network_output)
{
    if(max_y != NULL && min_y != NULL)
    {
        return network_output * (max_y - min_y) + min_y;
    }        
    else
    {
      return 0;
    }
}



// int main() {
//   vector<float> x_v = {7, 61, 47, 16, 51, 98, 15, 30, 19,  5, 16, 39, 56, 95, 73, 27, 93,
//        49, 38, 10, 81, 20, 18, 37, 82,  7, 53, 18, 84, 52, 38, 50, 47, 43,
//        77, 58, 48, 48, 24, 22, 70, 64, 23, 82, 47, 68, 95, 37, 62, 75,  3,
//        80, 76, 70, 80, 39, 12, 96, 78,  3, 52, 93, 52, 87,  9, 67,  2, 11,
//        13, 61,  2, 16, 49, 42, 45, 95, 93, 86, 83, 98, 54, 16, 20, 72,  2,
//        70,  5,  7, 31, 27,  8, 98, 54, 61, 49, 59, 31, 47, 82,  6};
//   vector<float> z_v = {
//         59, -41, -40,  71,  82, -61, -62,  64,  59,  60, -59, -69, -70,
//          5, -59, -34,  12,  -1,   8, -87,  55, -59,  44, -66,  96, -24,
//        -78,  59,  92, -10,   5, -85, -36, -43, -47,  36,  47, -68,  87,
//         84, -32,  19,  53, -59,  72,   1,  79, -88, -32, -89,  57,  54,
//        -74,  74, -10,  14,  26, -99,  44, -92,  73, -27,  -8, -55,  29,
//         29,  78,  34, -81, -60,  59, -37, -50, -79, -68, -62,   1, -47,
//         -1,  84, -82, -84, -69,  82, -73,   1, -28, -87,  36, -92,  55,
//         34,  89, -72, -62, -98, -78, -32,  -6, -18
//   };
//   vector<float> y_v = {4.10830881e+05, -1.33937594e+05, -1.25651024e+05,  7.16128001e+05,
//         1.10549069e+06, -4.44065260e+05, -4.76386454e+05,  5.25278608e+05,
//         4.11175229e+05,  4.32040076e+05, -4.10453766e+05, -6.55379265e+05,
//        -6.82696533e+05,  9.56127059e+03, -4.05208965e+05, -7.77981450e+04,
//         1.23845958e+04,  2.54559352e+03,  2.58398935e+03, -1.31687675e+06,
//         3.39554558e+05, -4.10298738e+05,  1.70745445e+05, -5.73512175e+05,
//         1.77644201e+06, -2.75780941e+04, -9.46135620e+05,  4.11135987e+05,
//         1.56468405e+06,  8.61302403e+02,  1.80910132e+03, -1.22559839e+06,
//        -9.09603192e+04, -1.57035627e+05, -2.01485038e+05,  9.68491700e+04,
//         2.10092804e+05, -6.26415772e+05,  1.31765278e+06,  1.18595852e+06,
//        -6.04246308e+04,  1.80041007e+04,  2.98350291e+05, -4.03788321e+05,
//         7.48845410e+05,  4.82970457e+03,  9.95387354e+05, -1.36146433e+06,
//        -6.15047082e+04, -1.40408815e+06,  3.70404261e+05,  3.21566783e+05,
//        -8.04443178e+05,  8.15558803e+05,  4.64016536e+03,  7.12606027e+03,
//         3.53315871e+04, -1.93109290e+06,  1.76685188e+05, -1.55735875e+06,
//         7.80894286e+05, -3.04375909e+04,  1.83485353e+03, -3.24919819e+05,
//         4.88851399e+04,  5.34665442e+04,  9.49114127e+05,  7.87607188e+04,
//        -1.06267519e+06, -4.28096291e+05,  4.10768083e+05, -1.01000805e+05,
//        -2.47451485e+05, -9.84187076e+05, -6.26705394e+05, -4.67347460e+05,
//         8.92985480e+03, -1.99991731e+05,  7.13464712e+03,  1.19530392e+06,
//        -1.09965888e+06, -1.18510454e+06, -6.56557635e+05,  1.10813565e+06,
//        -7.78024343e+05,  5.11092109e+03, -4.38639326e+04, -1.31693620e+06,
//         9.43659173e+04, -1.55656651e+06,  3.32838503e+05,  8.85054521e+04,
//         1.41301591e+06, -7.42593238e+05, -4.74106861e+05, -1.87872590e+06,
//        -9.48048827e+05, -6.31861664e+04,  6.53848264e+03, -1.16114217e+04
//   };

//   vector<float> x_test_v = {
//     41, 58, 70, 86, 10, 25, 25, 95, 69,  2, 91,  9, 40, 20, 34, 63, 93,
//        46, 87, 95, 86, 33, 53, 99, 64, 51, 74, 56, 62, 69, 76,  8, 68,  4,
//        31, 58, 41, 63, 91, 69, 61, 77, 65, 12, 43, 70, 39, 41, 79,  7, 15,
//        54, 78, 36,  8, 27, 82, 59, 73, 13, 82, 11, 26, 26, 57, 60, 77, 58,
//        88, 33, 48, 63, 27, 48, 19, 95, 72, 90, 16, 43, 90, 37, 50, 90, 97,
//        12, 39, 47, 11, 25, 61, 11, 60, 26,  1, 44, 10, 76, 80, 69
//   };
//   vector<float> z_test_v = {
//     -21,   67,  -84,  -33,  -93,   91,  -42,   18,  -41,   77,  -67,
//          48,   86,   73,   84,    7,  -84,   28,  -27,  -76,    9,  -65,
//         -47,  -36,   83,  -38,   76,  -55,    9,  -42,  -15,   29,   56,
//         -53,  -81,  -84,   15,   11,   57,  -91,  -10,   70,   45,   12,
//         -76,   23,   24,   62,   44,   36,  -54,  -58,    9,    8,  -96,
//          63,  -94,   88,  -85,   39,  -76,   90,  -19,  -74,  -21,  -62,
//          30,  -22,   -9,  -74,   35,   48,   37,  -94,  -66,   83,   15,
//           1,   89,   49,  -17,   72,   15,  -19,  -55,   28,  -24,    6,
//          24,  -81,   65,   96, -100,  -38,   38,   71,  -63,  -39,   93,
//          40
//   };
//   vector<float> y_test_v = {
//     -1.67179756e+04,  6.05064017e+05, -1.18029799e+06, -6.42199884e+04,
//        -1.60858390e+06,  1.50784204e+06, -1.47475960e+05,  2.09740105e+04,
//        -1.32873986e+05,  9.13076500e+05, -5.92971989e+05,  2.21292111e+05,
//         1.27383202e+06,  7.78494050e+05,  1.18666603e+06,  4.84401587e+03,
//        -1.17647999e+06,  4.61580217e+04, -3.15359885e+04, -8.68641989e+05,
//         9.11201163e+03, -5.48061970e+05, -2.04677981e+05, -8.32139899e+04,
//         1.14786202e+06, -1.06989980e+05,  8.83650014e+05, -3.29445982e+05,
//         5.48801613e+03, -1.43207986e+05, -7.45986842e+02,  4.88661250e+04,
//         3.56060015e+05, -2.97725750e+05, -1.06182797e+06, -1.18186998e+06,
//         8.55402439e+03,  6.82001587e+03,  3.78940011e+05, -1.50217399e+06,
//         1.90401639e+03,  6.92160013e+05,  1.86670015e+05,  3.63608333e+03,
//        -8.75973977e+05,  2.94440143e+04,  2.92860256e+04,  4.78460024e+05,
//         1.76846013e+05,  9.33821429e+04, -3.14657933e+05, -3.87145981e+05,
//         7.77601282e+03,  2.42802778e+03, -1.76938388e+06,  5.00904037e+05,
//        -1.65419799e+06,  1.36660202e+06, -1.22270199e+06,  1.18846077e+05,
//        -8.70981988e+05,  1.45815409e+06, -1.29639615e+04, -8.09693962e+05,
//        -1.51019825e+04, -4.72875983e+05,  6.01600130e+04, -1.77579828e+04,
//         6.55001136e+03, -8.09259970e+05,  8.81980208e+04,  2.25342016e+05,
//         1.02116037e+05, -1.65871998e+06, -5.74573947e+05,  1.15288401e+06,
//         1.21500139e+04,  8.37201111e+03,  1.41024206e+06,  2.37276023e+05,
//        -1.45598889e+03,  7.47976027e+05,  9.40002000e+03, -5.34798889e+03,
//        -3.23049990e+05,  4.40840833e+04, -2.60099744e+04,  2.78202128e+03,
//         2.78020909e+04, -1.06218196e+06,  5.53154016e+05,  1.76962609e+06,
//        -1.99621998e+06, -1.08989962e+05,  1.09749000e+05,  7.17890023e+05,
//        -4.99963900e+05, -1.12633987e+05,  1.61535401e+06,  1.32968014e+05
//   };


//   NetDDPRQNet net(1, 1);  

//   // prenorm
//   vector<float>::const_iterator y_min_iter, y_max_iter;
//   tie(y_min_iter, y_max_iter) = minmax_element(y_v.begin(), y_v.end());
//   min_y = *y_min_iter;
//   max_y = *y_max_iter;
//   vector<float>::const_iterator x_min_iter, x_max_iter;
//   tie(x_min_iter, x_max_iter) = minmax_element(x_v.begin(), x_v.end());
//   float min_x = *x_min_iter;
//   float max_x = *x_max_iter;
//   vector<float>::const_iterator z_min_iter, z_max_iter;
//   tie(z_min_iter, z_max_iter) = minmax_element(z_v.begin(), z_v.end());
//   float min_z = *z_min_iter;
//   float max_z = *z_max_iter;

//   cout << "min_y: " << min_y << "max_y: " << max_y << endl;
//   cout << "min_x: " << min_x << "max_x: " << max_x << endl;
//   cout << "min_z: " << min_z << "max_z: " << max_z << endl;

//   // normalization  
//   float *x = normalize_vec(x_v).data();
//   float *y = normalize_vec(y_v).data();
//   float *z = normalize_vec(z_v).data();
//   float *x_test = normalize_vec(x_test_v).data();
//   float *y_test = normalize_vec(y_test_v).data();
//   float *z_test = normalize_vec(z_test_v).data();

//   auto state = torch::from_blob(x, {100, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
//   auto action = torch::from_blob(z, {100, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
//   auto target = torch::from_blob(y, {100, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
//   // cout << "state: " << state << endl;
//   // cout << "action: " << action << endl;
  
//   auto optimizer = torch::optim::Adam(net->parameters(), torch::optim::AdamOptions(1e-3));
//   for(long long epoch = 0; epoch < 100000; epoch++)
//   {
//     optimizer.zero_grad();
//     torch::Tensor prediction = net->forward(state, action).reshape({100, 1});
//     torch::Tensor loss = torch::mse_loss(prediction, target);
//     loss.backward();
//     optimizer.step();
//     if(epoch % 10000 == 0)
//     {
//       cout << "epoch: " << epoch << " loss: " << loss.item<float>() << endl;
//     }
//   }
  
//   {
//     torch::NoGradGuard no_grad;
//     auto state_test = torch::from_blob(x_test, {100, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
//     auto action_test = torch::from_blob(z_test, {100, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
//     auto target_test = torch::from_blob(y_test, {100, 1}, torch::TensorOptions().dtype(torch::kFloat32)).clone();

//     auto test_prediction = net->forward(state_test, action_test);
//     auto test_loss = torch::mse_loss(test_prediction, target_test);
//     cout << "test loss: " << test_loss.mean().item<float>() << endl;
//   } 
//   torch::save(net, "net.pt");
// }


int main() {
  torch::Tensor x = torch::ones({1, 3}, torch::TensorOptions().requires_grad(true));
  cout << x << endl;
  x.data().mul_(2);
  cout << x << endl;
}
