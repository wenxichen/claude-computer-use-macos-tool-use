# Computer Use for tools on MacOS as a team

This is to use Anthropic's Computer Use to perform tasks on MacOS with multiple agents working together. The program runs the native MacOS. It uses Claude 3.5 Sonnet to perform tasks on your Mac by simulating mouse and keyboard actions as well as running bash command. Please use this with caution. 

This is built on top of the [Claude Computer Use Demo for MacOS](https://github.com/PallavAg/claude-computer-use-macos) which was originally forked from Anthropic's [computer use demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) - optimized for MacOS.

The multi-agent pipeline is described here: https://docs.google.com/presentation/d/1Ua2YGQyAPQ-lg7lIAFq_q6zhwuLkC0wy3wPt6LmXAu0/edit?usp=sharing

Some examples are shown in the notion notes here: https://resilient-rabbit-7f2.notion.site/Mac-Computer-Use-Tool-Use-Examples-13ae229b4ca080afb12bfed6600bf724

View Anthropic's docs [here](https://docs.anthropic.com/en/docs/build-with-claude/computer-use).

> [!CAUTION]
> - **Security Risks:** This script allows claude to control your computer's mouse and keyboard and run bash commands. Use it at your own risk.
> - **Responsibility:** By running this script, you assume all responsibility and liability for any results.
> - **AgentOps:** This script uses AgentOps to track the usage of the tool. If you do not want to track the usage, please do not set the `AGENTOPS_API_KEY` environment variable.
>
> Computer use is a beta feature. Please be aware that computer use poses unique risks that are distinct from standard API features or chat interfaces. These risks are heightened when using computer use to interact with the internet. To minimize risks, consider taking precautions such as:
>
> 1. Use a dedicated virtual machine or container with minimal privileges to prevent direct system attacks or accidents.
> 2. Avoid giving the model access to sensitive data, such as account login information, to prevent information theft.
> 3. Limit internet access to an allowlist of domains to reduce exposure to malicious content.
> 4. Ask a human to confirm decisions that may result in meaningful real-world consequences as well as any tasks requiring affirmative consent, such as accepting cookies, executing financial transactions, or agreeing to terms of service.
>
> In some circumstances, Claude will follow commands found in content even if it conflicts with the user's instructions. For example, instructions on webpages or contained in images may override user instructions or cause Claude to make mistakes. We suggest taking precautions to isolate Claude from sensitive data and actions to avoid risks related to prompt injection.
>
> Finally, please inform end users of relevant risks and obtain their consent prior to enabling computer use in your own products.
>
> ---
> This program focuses on using browser and web tools to perform tasks. One way to curb the risks is to limit the internet access to a whitelist of domains. You can do this by setting the allowed websites by following the steps below.
> 1. Open System Preferences
> 2. Select Screen Time
> 3. Select Content & Privacy
> 4. Toggle on Content & Privacy
> 5. Go to App Store, Media, Web, & Games
> 6. Click on Access to Web Content
> 7. Select Allowed Websites Only
> 8. Click on Customize to customize the allowed websites
> 9. Click Done

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

4. **[Optional] Set your AgentOps API key as an environment variable:**

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
