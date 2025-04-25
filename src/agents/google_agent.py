import google.generativeai as genai

class GoogleAgent:
    def __init__(self, apikey):
        if not apikey:
            raise Exception("Cannot intialize Google agent without API key.")

        self.ky = apikey
        genai.configure(api_key=self.ky)
        self.model = genai.GenerativeModel('models/gemini-2.0-flash')

    def ask(self, query, max_length=None):
        """
        Send a query to the Gemini LLM and return the response as a string.

        Args:
            query (str): The question or prompt to send to Gemini
            max_length (int, optional): Maximum length of the response in characters.
                                        If None, return the full response.

        Returns:
            str: The text response from Gemini, optionally truncated
        """
        try:
            generation_config = None
            if max_length is not None:
                # Note: This is an approximate conversion as tokens ≈ 4 characters
                # We divide by 4 to convert characters to approximate token count
                estimated_tokens = max(1, int(max_length / 4))
                generation_config = genai.GenerationConfig(max_output_tokens=estimated_tokens, min_output_tokens=20)

            response = self.model.generate_content(
                query,
                generation_config=generation_config
            )

            response_text = response.text
            if max_length is not None and len(response_text) > max_length:
                response_text = response_text[:max_length]

            return response_text
        except Exception as e:
            return f"Error querying Gemini: {str(e)}"
