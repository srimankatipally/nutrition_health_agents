# Nutrition Health Agents

This project is a personalized nutrition advisor application built using Streamlit. It Uses AI agents to provide tailored nutritional recommendations based on user input, including demographics, health conditions, and personal preferences.
Required OPENAI(LLM Usage) and SERPER(Tool for LLM Usage).
## Project Structure

```
nutrition_health_agents
├── nutrition_health_agent.py # Core functionality for the nutrition health agent
├── .env                    # Environment variables for API keys or if not you can enter in webapp
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/srimankatipally/nutrition_health_agents.git
   cd nutrition_health_agents
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SERPER_API_KEY=your_serper_api_key
   ```

## Usage

To run the application, execute the following command:
```bash
streamlit run nutrition_health_agent.py
```

Follow the prompts in the application to input your information and receive personalized nutrition recommendations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.
