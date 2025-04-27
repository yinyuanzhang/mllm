#include <iostream>
#include "cmdline.h"
#include "models/phi3v/configuration_phi3v.hpp"
#include "models/phi3v/modeling_phi3v.hpp"
#include "models/phi3v/processing_phi3v.hpp"
#include "processor/PostProcess.hpp"
#include <chrono>

using namespace mllm;
int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/phi3v_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/phi-3-vision-instruct-q4_k.mllm");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 2500);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    int thread_num = cmdParser.get<int>("thread");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto start_time = std::chrono::high_resolution_clock::now(); // 开始计时加载时间
    ParamLoader param_loader(model_path);
    auto processor = Phi3VProcessor(vocab_path);
    Phi3VConfig config(tokens_limit, "3.8B");
    auto model_config = Phi3VConfig(config);
    auto model = Phi3VModel(model_config);
    model.load(model_path);
    auto load_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Load time: " << load_time << " s" << std::endl;

    vector<string> in_imgs = {
        "../assets/australia.jpg"};
    vector<string> in_strs = {
        "<|image_1|>\nWhat's the content of the image?",
    };

    // 在进入循环之前定义变量
    std::chrono::high_resolution_clock::time_point data_preprocessing_start_time, data_preprocessing_end_time, prefill_start_time, first_token_time, decode_start_time;
    double ttft = 0.0;
    double total_decode_time = 0.0;
    int total_generated_tokens = 0;
    double data_preprocessin = 0.0;

    for (int i = 0; i < in_strs.size(); ++i) {
        data_preprocessing_start_time = std::chrono::high_resolution_clock::now();
        auto in_str = in_strs[i];
        in_str = processor.tokenizer->apply_chat_template(in_str);
        auto input_tensor = processor.process(in_str, in_imgs[i]);
        std::cout << "[Q] " << in_strs[i] << std::endl;
        std::cout << "[A] " << std::flush;
        data_preprocessing_end_time = std::chrono::high_resolution_clock::now();
        data_preprocessin = std::chrono::duration<double>(data_preprocessing_end_time - data_preprocessing_start_time).count();
        // 记录数据预处理的时间
        std::cout << "Data_preprocessin: " << data_preprocessin << " seconds" << std::endl;

        // 开始计时 TTFT
        prefill_start_time = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < 100; step++) {
            auto result = model(input_tensor);

            // 记录开始进行Model处理的时间
            std::time_t model_process = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << "model(input_tensor) time cost: " << std::ctime(&model_process); 

            // 记录生成第一个 token 的时间
            if (step == 0) {
                first_token_time = std::chrono::high_resolution_clock::now();
                ttft = std::chrono::duration<double>(first_token_time - prefill_start_time).count();
                std::cout << "TTFT: " << ttft << " seconds" << std::endl;
                // 开始计时解码阶段
                decode_start_time = first_token_time;
            }

            auto outputs = processor.detokenize(result[0]);
            auto out_string = outputs.first;
            auto out_token = outputs.second;
            auto [not_end, output_string] = processor.tokenizer->postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;

            // 更新生成的 token 数量
            total_generated_tokens += out_string.size(); // 根据解码的输出更新 token 计数

            chatPostProcessing(out_token, input_tensor[0], {&input_tensor[1], &input_tensor[2]});

            // 打印分界线
            std::cout << "-----------------------------------------------------------" << std::endl;
        }

        // 计算解码阶段的总时间
        auto decode_end_time = std::chrono::high_resolution_clock::now();
        total_decode_time = std::chrono::duration<double>(decode_end_time - decode_start_time).count();
        std::cout << "\nDecoding speed: " << total_generated_tokens / total_decode_time << " tokens/s" << std::endl;

        printf("\n");
    }

    return 0;
}