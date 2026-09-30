# Run the agent with Llama 3.1 on the Mac

You need: the Ollama app with `llama3.1` already installed, and the Python setup you already use for this project.

1. **Copy this whole `GEN-AI` folder to the Mac** (replace the old one).

2. **Open Terminal inside the folder**
   ```bash
   cd path/to/GEN-AI
   ```

3. **Check Ollama has the model** (open the Ollama app first if it is not running)
   ```bash
   ollama list
   ```
   `llama3.1` should be in the list.

4. **Tell the app to use it**
   ```bash
   printf "AVATARBLEND_LLM_PROVIDER=ollama\nAVATARBLEND_OLLAMA_MODEL=llama3.1\n" > .env
   ```

5. **Test the connection** (about 10 seconds)
   ```bash
   python3 check_llm.py
   ```
   You want three `OK` lines and `ready for the demo`.

6. **Run the app**
   ```bash
   streamlit run app.py
   ```
   The sidebar should show a green `LLM: ollama (llama3.1)`.

7. **Try it** in the "Describe it" tab: `curly red hair, heart sunglasses and a beard`, then `make the eyes blue`, then `remove the beard`, then `a friendly librarian`.

8. **Optional: measure Llama against the rule-based parser** (a few minutes)
   ```bash
   python3 evaluate_agent.py
   ```
   The results are written to `results/metrics_agent.json`.

## If something goes wrong
- `No module named ...`: run `pip3 install requests python-dotenv`.
- `model 'llama3.1' is not installed`: the name in `ollama list` is different; put that exact name in `.env`.
- `FELL BACK TO RULES` or `PROBLEM`: the app still works using the rule-based fallback, so you can still demo it.
