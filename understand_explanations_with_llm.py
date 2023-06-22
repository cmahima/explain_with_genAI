import warnings


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv, find_dotenv

def get_explanation_from_genAI(shap_input, pred, problem):
    warnings.filterwarnings('ignore')
    _ = load_dotenv(find_dotenv()) # read local .env file

    # To control the randomness and creativity of the generated
    # text by an LLM, use temperature = 0.0
    chat = ChatOpenAI(temperature=0.0)

    template_string_iris = """ For the following text, extract the following information:
    The input is a map:{input_map}, where the key is string and value is a decimal value which represnts the Shapley value of the feature.
    The prediction is in {pred}. If prediction is 0, the class is Iris-setosa; if prediction is 1, the class is Iris Versicolor; If prediction is 2 class is Iris Virginica.
    Output the summary of feature impact based on shapley values in layman terms.
    The output should denote the impact(bold font) of each feature in bullet points only with their shapley values.
    """
    #what would I do to increase shapley value for a feature
    template_string_flights = """ For the following text, extract the following information:
    The input is a map:{input_map}, where the key is string and value is a decimal value which represnts the Shapley value of the feature for regression problem.
    The prediction is in {pred}. The prediction denotes the late if postive, early if negative or on time if zero.
    Output the summary of feature impact based on shapley values in layman terms.
    The output should denote the impact(bold font) of each feature in bullet points only with their shapley values.
    """

    if "Classification" in problem:
        prompt_template = ChatPromptTemplate.from_template(template_string_iris)
    else:
        prompt_template = ChatPromptTemplate.from_template(template_string_flights)

    messages = prompt_template.format_messages(
                        input_map=shap_input, pred=pred)
    # Call the LLM to translate to the style of the readble message
    response = chat(messages)
    print(messages)
    return response.content
