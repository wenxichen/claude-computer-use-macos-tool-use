# Computer Use for tools on MacOS as a team

This is to use Anthropic's Computer Use to perform tasks on MacOS with multiple agents working together. The program runs the native MacOS. It uses Claude 3.5 Sonnet to perform tasks on your Mac by simulating mouse and keyboard actions as well as running bash command. Please use this with caution. 

This is built on top of the [Claude Computer Use Demo for MacOS](https://github.com/PallavAg/claude-computer-use-macos) which was originally forked from Anthropic's [computer use demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) - optimized for MacOS.

The multi-agent pipeline is described here: https://docs.google.com/presentation/d/1Ua2YGQyAPQ-lg7lIAFq_q6zhwuLkC0wy3wPt6LmXAu0/edit?usp=sharing

Some examples are shown in the notion notes here: https://resilient-rabbit-7f2.notion.site/Mac-Computer-Use-Tool-Use-Examples-13ae229b4ca080afb12bfed6600bf724

View Anthropic's docs [here](https://docs.anthropic.com/en/docs/build-with-claude/computer-use).

> [!WARNING]  
> Use this script with caution. Allowing Claude to control your computer can be risky. By running this script, you assume all responsibility and liability.

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/wenxichen/claude-computer-use-macos-tool-use.git
   cd claude-computer-use-macos-tool-use
   ```

2. **Create a virtual environment + install dependencies:**

   ```bash
   conda create -n computer-use python=3.13
   conda activate computer-use
   pip install -r requirements.txt
   ```

3. **Set your Anthropic API key as an environment variable:**

   ```bash
   export ANTHROPIC_API_KEY="CLAUDE_API_KEY"
   ```

   Replace `CLAUDE_API_KEY` with your actual Anthropic API key. You find yours [here](https://console.anthropic.com/settings/keys).

4. **Set your optional AgentOps API key as an environment variable:**

 ```bash
   export AGENTOPS_API_KEY="AGENTOPS_API_KEY"
   ```

   Replace `AGENTOPS_API_KEY` with your actual AgentOps API key. You find yours [here](https://app.agentops.ai/settings/projects).

5. **Grant Accessibility Permissions:**

   The script uses `pyautogui` to control mouse and keyboard events. On MacOS, you need to grant accessibility permissions. These popups should show automatically the first time you run the script so you can skip this step. But to manually provide permissions:

   - Go to **System Preferences** > **Security & Privacy** > **Privacy** tab.
   - Select **Accessibility** from the list on the left.
   - Add your terminal application or Python interpreter to the list of allowed apps.

## Usage

You can run the script by passing the instruction directly via the command line or by editing the `main.py` file.

**Example using command line instruction:**

```bash
python main.py 'Open Safari and look up Anthropic'
```

Replace `'Open Safari and look up Anthropic'` with your desired instruction.

## Exiting the Script

You can quit the script at any time by pressing `Ctrl+C` in the terminal.

## ⚠ Disclaimer

> [!CAUTION]
> - **Security Risks:** This script allows claude to control your computer's mouse and keyboard and run bash commands. Use it at your own risk.
> - **Responsibility:** By running this script, you assume all responsibility and liability for any results.
