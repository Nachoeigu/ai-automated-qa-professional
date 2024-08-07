from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
import pandas as pd
from langchain.prompts import PromptTemplate

#... Import the rest of dependencies to solve the problem


articles = {
    "New Political Agreement Aims to Enhance Global Cooperation": "A recently brokered political agreement among several key nations aims to enhance global cooperation on various critical issues, including security, trade, and environmental protection. This pact signifies a major step towards unified international policies and demonstrates a commitment to addressing global challenges through collaborative efforts. Observers are optimistic that this agreement will lead to more stable and prosperous international relations.",
    "Historic Victory in Major Sports Tournament Stuns Fans": "In an unexpected turn of events, an underdog team has claimed victory in a major sports tournament, stunning fans and experts alike. The team's remarkable performance throughout the competition has been lauded as one of the greatest upsets in sports history. This win is expected to inspire a new generation of athletes and highlights the unpredictable and thrilling nature of sports.",
    "Climate Change Report Warns of Accelerating Environmental Impact": "A new climate change report warns that the environmental impact of global warming is accelerating at an alarming rate. The study highlights the urgent need for immediate and comprehensive action to mitigate the effects of climate change, including rising sea levels, extreme weather events, and loss of biodiversity. Experts call for stronger international policies and increased investment in sustainable practices to address this critical issue.",
    "Economic Forecast Predicts Growth Amidst Global Challenges": "Despite ongoing global challenges, a recent economic forecast predicts moderate growth for the coming year. Factors contributing to this optimistic outlook include technological advancements, increased consumer spending, and resilient supply chains. However, the report also cautions about potential risks such as geopolitical tensions and inflation, emphasizing the need for adaptive economic policies to sustain growth."
}

possible_categories = [
    "politics",
    "economics",
    "sports",
    "society",
    "insecurity",
    "enviroment"
]

from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field

class StructuringLLMOutput(BaseModel):
    """Structuring the LLM output in a Pydantic object"""
    article_category: Literal["politics","economics","sports","society","insecurity","enviroment"] = Field(..., description="Select the category that represents better the news article.")

class NewspaperClassifier:

    def __init__(self, model):
        self.model = model.with_structured_output(StructuringLLMOutput)
        self.__developing_template() 
        self.chain = self.__developing_chain()

    
    def __developing_template(self):
        """
        This method creates a prompt template that will be pass through the LLM model
        """
        self.prompt_template = PromptTemplate(
                template="You are an expert news article classifier, who analyzes meticulously its content in order to determinate its category.\nClassify the following one: `{article_content}´",
                input_variables=["article_content"],
            )
    def __developing_chain(self):
        """
        This method instances the entire chain the classifier will utilize in each classification
        """
        return self.prompt_template | self.model

if __name__ == '__main__':
    model = ChatOpenAI(model="gpt-4o-mini", 
                    temperature = 0)
    news_classifier = NewspaperClassifier(model = model)

    df = pd.DataFrame(columns = ['article_title','article_content','article_category'])

    for article in articles.items():
        # We place the title (dict key) and content (dict value) in one string variable
        article_text = "Title:\n" + article[0] + '\n\n' + "Content:\n" + article[1]
        output = news_classifier.chain.invoke({"article_content": article_text})

        print(f"For the article: {article[0]}, the model predicts: {output.article_category}")
        
        df = pd.concat([df, pd.DataFrame({"article_title":article[0], "article_content": article[1], "article_category": output.article_category}, index = [0])])


    df.to_csv("all_articles_classified.csv", index = False)

