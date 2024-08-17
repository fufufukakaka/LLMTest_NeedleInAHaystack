import os

from langchain.evaluation import load_evaluator
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, Field

from .evaluator import Evaluator


class AnthropicEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    SINGLE_NEEDLE_CRITERIA = {
        "accuracy": """
            スコア 1: 回答が参照情報と全く関連がない。もしくは抽出することができなかった。
            スコア 3: 回答には若干の関連性があるが、参照情報と一致しない。
            スコア 5: 回答には適度な関連性があるが、不正確な情報が含まれている。
            スコア 7: 回答が参照情報と一致しているが、若干の欠落がある。
            スコア 9: 回答が参照情報と一致しているが、非常に些細な欠落がある。
            スコア 10: 回答が完全に正確で、参照情報と完全に一致している。
            数値のスコアのみを回答してください。"""
    }

    def __init__(
        self,
        model_name: str = "claude-3-opus-20240229",
        model_kwargs: dict = DEFAULT_MODEL_KWARGS,
        true_answer: str = None,
        question_asked: str = None,
    ):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError(
                "true_answer and question_asked must be supplied with init."
            )

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv("NIAH_EVALUATOR_API_KEY")
        if not api_key:
            raise ValueError(
                "NIAH_EVALUATOR_API_KEY must be in env for using anthropic evaluator."
            )

        self.api_key = api_key

        self.evaluator = ChatAnthropic(
            model=self.model_name, anthropic_api_key=api_key, **self.model_kwargs
        )

    def evaluate_response(
        self,
        response: str,
        multi_needles: bool = False,
        evaluate_fake_answer: bool = False,
    ) -> int:
        if multi_needles:
            eval_result = self.evaluate_responses_in_multi_needles(
                student_answer=response, reference=self.true_answer
            )
            return int(eval_result)
        if evaluate_fake_answer:
            eval_result = self.evaluate_responses_in_multi_needles(
                student_answer=response, reference=self.fake_answer
            )
            return int(eval_result)
        else:
            evaluator = load_evaluator(
                "labeled_score_string",
                criteria=self.CRITERIA,
                llm=self.evaluator,
            )

            eval_result = evaluator.evaluate_strings(
                # The models response
                prediction=response,
                # The actual answer
                reference=self.true_answer,
                # The question asked
                input=self.question_asked,
            )

            return int(eval_result["score"])

    def evaluate_responses_in_multi_needles(
        self, student_answer, reference
    ) -> int:
        """
        multi needle evaluation ではいくつの針を抽出できたかを評価する。
        """

        # Grade prompt
        template = """あなたは参照情報を見ながら、生徒の回答に対して採点を行うことが得意な採点者です。
            参照情報は2文章以上から構成されています。生徒の回答には、参照情報に含まれる情報を正確に抽出することが求められます。
            生徒の回答はいくつかの文章から構成されています。生徒の回答文の中で参照情報の情報の数と一致する数を求めることがあなたの仕事です。
            例えば、参照情報に5つの情報が含まれているとします。これに対して生徒の回答が5つの情報を含んでおり、それらが参照情報と一致しているのであればスコアは5となります。
            一方で、生徒の回答が5つの情報を含んでいるが、そのうち2つが参照情報と一致していない場合、スコアは3となります。
            注意深く生徒の回答を見て、参照情報との一致度を評価してください。
            例えば、参照情報が「A,B」だったときに、生徒の回答が「A」だった場合、スコアは1となります。参照情報が「今期は予算カツカツで厳しいです, 500万円から経営会議にかけなきゃだめなんですよ, 導入すればROIは立つとは思っています。」だったときに、生徒の回答が「今期は予算カツカツで厳しいです, 500万円から経営会議にかけなきゃだめなんですよ」だった場合、スコアは2となります。
            参照情報が「今期は予算カツカツで厳しいです, 500万円から経営会議にかけなきゃだめなんですよ, 導入すればROIは立つとは思っています。100万円であれば問題ないと思います」だったときに、生徒の回答が「今期は予算カツカツで厳しいです, 100万円であれば問題ないと思います」だった場合、スコアは2となります。
            参照情報が「今期は予算カツカツで厳しいです, 500万円から経営会議にかけなきゃだめなんですよ, 導入すればROIは立つとは思っています。100万円であれば問題ないと思います」だったときに、生徒の回答が「今期は予算カツカツで厳しいです, 500万円から経営会議にかけなきゃだめなんですよ, 導入すればROIは立つとは思っています。100万円であれば問題ないと思います」だった場合、スコアは4となります。
            注意深く生徒の回答を見て、参照情報との一致度を評価してください。
            生徒の回答: \n --- --- --- \n {answer}
            参照情報: \n --- --- --- \n {reference}"""
        # Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["answer", "reference"],
        )

        # Data model
        class grade(BaseModel):
            """Grade output"""

            score: int = Field(description="Score from grader")

        structured_llm = self.evaluator.with_structured_output(grade)

        chain = prompt | structured_llm

        score = chain.invoke({"answer": student_answer, "reference": reference}).score

        return score
