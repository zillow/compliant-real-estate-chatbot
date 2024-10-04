DEEPEVAL_RESULTS_FOLDER="data/geval_results/" \
OPENAI_API_KEY="your api key here"\
 python geval_evaluation.py --test_data_path path_to_generated_responses_by_some_model.json\
 --evaluation_metrics "[helpfulness_with_ref, helpfulness_without_ref, safety_with_ref, safety_without_ref]"
