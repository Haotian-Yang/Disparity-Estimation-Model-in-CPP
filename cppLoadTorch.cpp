#include <torch/script.h> // One-stop header.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <memory>
#include <opencv2/imgproc.hpp>

torch::Tensor matToTensor(cv::Mat const& src)
{
   	torch::Tensor out;
	cv::Mat img_float;
	src.convertTo(img_float, CV_32F, 1.0f/256.0f);
	std::cout << img_float << std::endl;
	
	out = torch::from_blob(img_float.data, {1, img_float.rows, img_float.cols, img_float.channels()});
    // H, W, C --> C, H, W
	out = out.permute({0, 3, 1, 2});
   	return out;
}

// MODIFIES original tensor so that theres no need of clone()
cv::Mat tensorToMat(torch::Tensor t)
{

	t = t.permute({1, 2, 0}).to(torch::kU8);
    //t = t.mul(255).clamp(0, 255).to(torch::kU8);
    //t = t.to(torch::kF32).mul(256);
    t = t.to(torch::kCPU);
	std::cout << t << std::endl;
    std::cout << t.size(0) << t.size(1)<< t.size(2) <<std::endl;
    cv::Mat resultImg(t.size(0), t.size(1), CV_8U);
	std::cout << resultImg.size() << std::endl;
    std::memcpy((void *) resultImg.data, t.data_ptr(), sizeof(torch::kU8) * t.numel());
	return resultImg;

}

int main(int argc, const char* argv[]) {
    //if (argc != 2) {
    //    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    //   return -1;
    //}

    // load model
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);//, torch::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }


    // load image
    cv::Mat left, right, disp;
    left = cv::imread(argv[2], cv::IMREAD_COLOR);   // Read the file
	
    //std::cout << left << std::endl;
    right = cv::imread(argv[3], cv::IMREAD_COLOR);   // Read the file
    torch::Tensor left_t = matToTensor(left).to(torch::Device(torch::kCUDA));
    torch::Tensor right_t = matToTensor(right).to(torch::Device(torch::kCUDA));
    //std::cout << left_t << std::endl;
    //std::cout << right_t << std::endl;

    std::cout << left_t.size(0) << left_t.size(1)<< left_t.size(2) << left_t.size(3) <<std::endl;
    std::cout << right_t.size(0) << right_t.size(1)<< right_t.size(2) << right_t.size(3) <<std::endl;
    if(! left.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(left_t);
    inputs.push_back(right_t);
    
    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    //std::cout << output.size(0) << output.size(1)<< output.size(2) <<std::endl;
    disp = tensorToMat(output);
    //std::cout << disp << std::endl;
	//cv::applyColorMap(disp, disp, cv::COLORMAP_JET);
    
    cv::imshow( "Display window", disp );                   // Show our image inside it.

    cv::waitKey(0);

    std::cout << "ok\n";
}
