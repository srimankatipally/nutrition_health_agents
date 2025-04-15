# Nutrition Health Agents

This project is a personalized nutrition advisor application built using Streamlit. It leverages AI agents to provide tailored nutritional recommendations based on user input, including demographics, health conditions, and personal preferences.

## Project Structure

```
nutrition_health_agents
├── nutrition_health_agent.py # Core functionality for the nutrition health agent
├── .env                    # Environment variables for API keys
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
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

## License

This project is licensed under the MIT License. See the LICENSE file for more details.