import requests
from time import sleep
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import langchain
from langchain_core.language_models.llms import LLM
from yandex_chain.util import YAuth, YException
from tenacity import Retrying, RetryError, stop_after_attempt, wait_fixed

class YandexLLM(LLM):
    async_retries: int = 5
    async_sleep_interval: float = 0.1
    temperature : float = 1.0
    max_tokens : int = 1500
    sleep_interval : float = 1.0
    retries = 3
    use_lite = True
    use_async = False
    instruction_text : str = None
    instruction_id : str = None
    iam_token : str = None
    folder_id : str = None
    api_key : str = None
    config : str = None

    inputTextTokens : int = 0
    completionTokens : int = 0
    totalTokens : int = 0
    disable_logging : bool = False

    @property
    def _llm_type(self) -> str:
        return "YandexGPT"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return { "max_tokens": self.max_tokens, "temperature" : self.temperature }

    @property
    def _modelUri(self):
        if self.instruction_id:
            return f"ds://{self.instruction_id}"
        if self.use_lite:
            return f"gpt://{self.folder_id}/yandexgpt-lite/latest"
        else:
            return f"gpt://{self.folder_id}/yandexgpt/latest"

    @staticmethod
    def UserMessage(message):
        return { "role" : "user", "text" : message }

    @staticmethod
    def AssistantMessage(message):
        return { "role" : "assistant", "text" : message }

    @staticmethod
    def SystemMessage(message):
        return { "role" : "system", "text" : message }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        return_message = False
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        msgs = []
        if self.instruction_text:
            msgs.append(self.SystemMessage(self.instruction_text))
        msgs.append(self.UserMessage(prompt))
        return self._generate_messages(msgs)

    def _generate_messages(self,
        messages : List[Mapping],       
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        return_message = False):
        auth = YAuth.from_params(dict(self))
        if not self.folder_id:
            self.folder_id = auth.folder_id
        req = {
          "modelUri": self._modelUri,
          "completionOptions": {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
          },
          "messages" : messages
        }
        headers = auth.headers
        if self.disable_logging:
            headers['x-data-logging-enabled'] = 'false'
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.retries),wait=wait_fixed(self.sleep_interval)):
                with attempt:
                    if self.use_async is False:
                        return self._send_sync(headers, req, return_message)
                    else:
                        return self._send_async(headers, req, return_message)
        except RetryError:
            raise YException(f"Error calling YandexGPT after {self.retries} retries.")

    def _send_sync(self, headers, req, return_message):
        res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers, json=req)
        js = res.json()
        if 'result' in js:
            res = js['result']
            return self._parse_result(res, return_message)
        raise YException(f"Cannot process YandexGPT request, result received: {js}")

    def _send_async(self, headers, req, return_message):
        res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync",
            headers=headers, json=req)
        js = res.json()
        if 'id' in js:
            operation_id = js['id']
        else:
            raise YException(f"Cannot process YandexGPT async request, result received: {js}")
        for _ in range(self.async_retries):
            sleep(self.async_sleep_interval)
            res = requests.get(f"https://llm.api.cloud.yandex.net/operations/{operation_id}",
                headers=headers, json=req)
            js = res.json()
            if js['done'] is True:
                response = js.get('response')
                if response is None:
                    raise YException(f"An error occurred while generating the response. Result returned: {js}")
                else:
                    return self._parse_result(response, return_message)

    def _parse_result(self, res, return_message):
        usage = res['usage']
        self.totalTokens += int(usage['totalTokens'])
        self.completionTokens += int(usage['completionTokens'])
        self.inputTextTokens += int(usage['inputTextTokens'])
        return res['alternatives'][0]['message'] if return_message else res['alternatives'][0]['message']['text']

    def resetUsage(self):
        self.totalTokens = 0
        self.completionTokens = 0
        self.inputTextTokens = 0
