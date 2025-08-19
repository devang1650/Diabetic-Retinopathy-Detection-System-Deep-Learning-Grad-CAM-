import google.generativeai as genai
from config import GEMINI_API_KEY
import warnings


class AIHelper:
    def __init__(self):
        try:
            # Suppress TensorFlow warnings that might come from dependencies
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')

            # Configure Gemini
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('models/gemini-1.5-pro-002')
        except Exception as e:
            warnings.warn(f"AI initialization failed: {str(e)}")
            raise

    def get_dr_analysis(self, severity):
        prompt = f"""As an ophthalmology AI analyzing {severity} diabetic retinopathy:

        Provide:
        1. Condition explanation (layman's terms)
        2. Recommended clinical actions (bulleted)
        3. Treatment options
        4. Prognosis
        5. Lifestyle recommendations

        Be concise (300 words max), use markdown formatting with **bold** headers."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ AI service unavailable\nError: {str(e)}"


# Fallback for when API key is not available
class DummyAIHelper:
    def get_dr_analysis(self, severity):
        return "AI analysis feature is currently unavailable"


if __name__ == "__main__":
    # Test the AI helper
    try:
        ai = AIHelper()
        print(ai.get_dr_analysis("Moderate"))
    except:
        ai = DummyAIHelper()
        print(ai.get_dr_analysis("Moderate"))