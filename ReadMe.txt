Dataset and Labels:
    Elevater_data: "/data/user4/CWW_Datasets/Elevater_data/classification"
    Elevater_base_novel_label: "/data/user4/cww/Data_Construction/fsl_data/elevater_label_new"
    FSL_data: "/data/user4/CWW_Datasets/FSL" (CIFAR-FS, MINI, TIERED)
    FSL_base_novel_label: "/data/user4/cww/Data_Construction/fsl_data/label"


Step1 Building Meta-task insturctions (Qwen-VL fine-tuning format datas):
    Folder: "build_data_instruction"
        1.The directory for building Qwen fine-tuning/test format in this folder is "building_Qwen_fine-tuning-format" and the source directory comes from "/data/user4/cww/Data_Construction/building_Qwen_fine-tuning-format".
        2.sample file:"sample_base_novel_ele_qwen.py".
        3.The well-formatted fine-tuning file is in "/data/user4/cww/Data_Construction/output7_newformat_Qwen_finetune_13_elevater_add_answerlist_QandA/output7_newformat_Qwen_finetune_13_elevater_add_answerlist_QandA_base_get_ratio_data.json".
        4.The well-formatted test files used before are stored in 'novel_5000balance' and 'novel_5000_balance_top2nomatch_filter'(The json directory after reconstructing the instruction data when the answer is not in the candidate in step 6) directory.


Step2 Finetuning Qwen:
    Folder "/data/user4/HUOJIAN/Qwen-VL/finetune"
        1.Due to GPU memory limitations, only the Qwen-VL-Chat-Int4 model can be used and only Qlora fine-tuning can be performed.
        2.The main fine-tuning script used: "/data/user4/cww/MLVLM-FSL/Finetune_Qwen_scrips/finetune_qlora_ds_cww.sh".
        3.In the script, make sure you are using the "Qwen-VL-Chat-Int4" model and confirm your "DATA" path (the json path of the fine-tuning data) and the GPU number you want to use, then execute this bash file in the terminal.


Step3 Testing Qwen
    Folder "Test_Qwen_scripts"
        1.In script, You can modify the "evaluate_config_path" to test different datasets. Previous eval_config is stored in "/data/user4/HUOJIAN/Qwen-VL/eval_mm/FSL_eval_config".
        2.You can modify the "chcekpoint" to test different models:
            a.For Original checkpoint without fine-tuning, execute the eva_notune_infer_fsl4_test.sh script and then it will run the evaluate_for_FSL_notune.py code file.
            b.For Qwen-VL with fine-tuning, execute the eva_ele_vanilia_3500_infer_fsl4_test_no_history.sh script and then it will run the evaluate_for_FSL_v1_no_history.py code file.
        3.The previous test results can be seen in "/data/user4/HUOJIAN/Qwen-VL/cww_test/" or "/data/user4/HUOJIAN/Qwen-VL/eval_mm/eval_results(the latter is more early,look the first is enough)".


Step4 The model output is post-processed using CLIP:
    File "cal_for_Qwen_using_CLIP.py"
        1.Given the data set name and a partial path to the file output from the model, you can choose CLIP-L-14 or CLIP-Big-G.
        2.Statistics of "Acc_clip"(the answer after the CLIP calculates the text similarity) and "Acc_occur"(the correct answer will occur as long as the correct answer appears in the output of Qwen-VL).


Step5 Building attribute descriptions:
    Folder "attribute_text"
        1.The py files starting with "mllm_generate_text" are all attribute text generation files for their respective datasets.
        2.The prompt text input to LVLM for different attribute descriptions is explained at the beginning of the py file, or can be viewed in the main text and appendix of the paper.
        3.The improved v3 version "mllm_generate_text_v3.py" can also be used.


Step6 Filter candidate answers based on model output results and attribute descriptions
    Folder "attribute_filter".
        1.The number is already labeled and executed in sequence. There are notes in the file.
        2.Calculate the attribute description similarity of support sample and query sample in attribute i of each test instruction, and obtain the aggregated similarity order (1_cal_aggregate_similarity). If the output answer of the model is in the top2 of the aggregate candidate answer set, it is considered reliable; if it is not, it is considered unreliable. Extract these unreliable instructions (3-chose_no_match_file) and rearrange them into new 3way instructions for the model to answer again (4-extract_no_match_to_json).



