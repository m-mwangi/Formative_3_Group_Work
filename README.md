# Formative_3_Group_Work



### **Team Onboarding & Experiment Guide** 🚀

We have now finalized the core training and playing scripts.

Your mission is to get your local environment set up, verify it works.

### **Step 1: Get the Latest Code**

First, make sure you have the most recent versions of all the project files.

```bash
# If you haven't cloned the project yet:
git clone [YOUR_REPO_URL]
cd [REPO_NAME]

# If you have already cloned it:
git pull origin main
```

-----

### **Step 2: Create Your Virtual Environment**

We must all use the *exact same* Python version and libraries.

1.  **Check Python:** Make sure you have a stable version of Python installed (e.g., **3.10** or **3.11**).
2.  **Create venv:** In your terminal, from the project's root folder, create your own local virtual environment.
    ```bash
    # On macOS / Linux
    python3 -m venv venv

    # On Windows
    python -m venv .venv
    ```
3.  **Activate venv:** You must do this **every time** you work on the project.
    ```bash
    # On macOS / Linux
    source .venv/bin/activate

    # On Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1

    # On Windows (Command Prompt)
    .\.venv\Scripts\activate.bat
    ```
    Your terminal prompt should now start with `(venv)`.

-----

### **Step 3: Install All Dependencies**

Now that your venv is active, install all the *exact* library versions from the project file.

```bash
# This command reads the requirements.txt file and installs everything
pip install -r requirements.txt
```

-----

### **Step 4: CRITICAL VERIFICATION**

This is the most important step. We must confirm your setup is 100% working before you run any long experiments.

**Test 1: Run a short training job.**
This command will train for only 1000 steps.

```bash
python train.py --steps 1000
```

  * **What to expect:** It should run, show a progress bar, and finish in 1-2 minutes. It will save a new `.zip` file in the `models/` folder.

**Test 2: Run the `play.py` script.**
This confirms you can watch the models you train.

1.  **Find your new model:**
    ```bash
    ls models/
    ```
2.  **Run `play.py` on that model.** Use the **Tab** key to auto-complete the long filename.
    ```bash
    # Type 'python play.py --model models/pol' and then press Tab
    python play.py --model "models/YOUR-MODEL-NAME.zip"
    ```

<!-- end list -->

  * **What to expect:** A game window for `Breakout` should pop up and you'll see the agent (playing badly) for 5 episodes.

> **If both tests pass, you are ready\!** If you get *any* error, stop and ask the group for help.
