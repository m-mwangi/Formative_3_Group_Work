# Formative_3_Group_Work



### **Team Onboarding & Experiment Guide** 🚀

We have now finalized the core training and playing scripts.

Your mission is to get your local environment set up, verify it works.

### **Step 1: Get the Latest Code**

First, make sure you have the most recent versions of all the project files.

```bash
# If you haven't cloned the project yet:
git clone https://github.com/m-mwangi/Formative_3_Group_Work.git
cd Formative_3_Group_Work

# If you have already cloned it:
git checkout script-setup
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


Here is the 40-experiment strategic plan, breaking down the tables and the reasoning for each group.

### 🎯 The Overall Strategy

The plan is designed as a **systematic "grid search,"** where each team member becomes a "specialist" isolating a specific set of variables. This is far better than random guessing, as it will allow you to see the direct impact of each hyperparameter.

* **Baseline (Control):** `lr=0.0001`, `gamma=0.99`, `batch_size=32`, `epsilon_decay=250000`, `epsilon_end=0.05`. Many runs will use these defaults to isolate other variables.

---

### 👨‍🔬 Member 1: "Learning Rate & Stability" Specialist

**Reasoning:** This member's mission is to find the **single most important hyperparameter: the learning rate (`lr`)**. A `lr` that is too high will be unstable and fail to learn. A `lr` that is too low will learn too slowly. This group also tests how `batch_size` interacts with the learning rate, as larger batches often allow for higher learning rates.

| Exp | Goal | `lr` | `batch_size` | Command to Run (`--steps 1000000`)... |
| :--- | :--- | :--- | :--- | :--- |
| **1.1** | Very Fast (Unstable?) | `0.001` | 32 | `--lr 0.001` |
| **1.2** | Fast | `0.0005` | 32 | `--lr 0.0005` |
| **1.3** | Slightly Fast | `0.00025` | 32 | `--lr 0.00025` |
| **1.4** | **Baseline** | `0.0001` | 32 | ` ` (default) |
| **1.5** | Slow | `0.00005` | 32 | `--lr 0.00005` |
| **1.6** | Very Slow | `0.00001` | 32 | `--lr 0.00001` |
| **1.7** | Large Batch | `0.0001` | `64` | `--batch_size 64` |
| **1.8** | Larger Batch | `0.0001` | `128` | `--batch_size 128` |
| **1.9** | *Interact:* Fast `lr` + Large Batch | `0.0005` | `128` | `--lr 0.0005 --batch_size 128` |
| **1.10** | *Interact:* Slow `lr` + Small Batch | `0.00005` | `16` | `--lr 0.00005 --batch_size 16` |

---

### 👨‍🔬 Member 2: "Exploration & Curiosity" Specialist

**Reasoning:** This member's mission is to find the **best exploration strategy**. How long should the agent be "curious" (controlled by `epsilon_decay`)? How much curiosity should it keep at the end (`epsilon_end`)? If it stops exploring too soon, it may never learn the best strategies. If it explores for too long, it may never stabilize and get a high score.

| Exp | Goal | `epsilon_decay` | `epsilon_end` | Command to Run (`--steps 1000000`)... |
| :--- | :--- | :--- | :--- | :--- |
| **2.1** | Very Fast Decay | `100000` | 0.05 | `--epsilon_decay 100000` |
| **2.2** | **Baseline** | `250000` | 0.05 | ` ` (default) |
| **2.3** | Slow Decay | `500000` | 0.05 | `--epsilon_decay 500000` |
| **2.4** | Very Slow Decay | `750000` | 0.05 | `--epsilon_decay 750000` |
| **2.5** | *Interact:* Fast Decay + No Final | `100000` | `0.01` | `--epsilon_decay 100000 --epsilon_end 0.01` |
| **2.6** | *Interact:* Slow Decay + High Final | `500000` | `0.1` | `--epsilon_decay 500000 --epsilon_end 0.1` |
| **2.7** | No Final Exploration | `250000` | `0.001` | `--epsilon_end 0.001` |
| **2.8** | Low-Mid Final Expl. | `250000` | `0.025` | `--epsilon_end 0.025` |
| **2.9** | High Final Expl. | `250000` | `0.1` | `--epsilon_end 0.1` |
| **2.10** | Very High Final Expl. | `250000` | `0.15` | `--epsilon_end 0.15` |

---

### 👨‍🔬 Member 3: "Long-Term Vision" Specialist

**Reasoning:** This member's mission is to find the **optimal discount factor (`gamma`)**. `gamma` controls how much the agent values future rewards. A low `gamma` makes the agent "short-sighted" (it only cares about immediate points). A high `gamma` makes it "long-sighted" (it's willing to sacrifice a point now to get 10 points later). This group also tests how `gamma` interacts with the learning rate.

| Exp | Goal | `gamma` | `lr` | Command to Run (`--steps 1000000`)... |
| :--- | :--- | :--- | :--- | :--- |
| **3.1** | Short-Sighted | `0.9` | 0.0001 | `--gamma 0.9` |
| **3.2** | Mid-Sighted | `0.95` | 0.0001 | `--gamma 0.95` |
| **3.3** | Standard | `0.98` | 0.0001 | `--gamma 0.98` |
| **3.4** | **Baseline** | `0.99` | 0.0001 | ` ` (default) |
| **3.5** | Very Long-Sighted | `0.995` | 0.0001 | `--gamma 0.995` |
| **3.6** | *Interact:* Fast `lr` + Short `gamma` | `0.95` | `0.0005` | `--gamma 0.95 --lr 0.0005` |
| **3.7** | *Interact:* Fast `lr` + Long `gamma` | `0.99` | `0.0005` | `--gamma 0.99 --lr 0.0005` |
| **3.8** | *Interact:* Slow `lr` + Short `gamma` | `0.95` | `0.00005` | `--gamma 0.95 --lr 0.00005` |
| **3.9** | *Interact:* Slow `lr` + Long `gamma` | `0.99` | `0.00005` | `--gamma 0.99 --lr 0.00005` |
| **3.10** | *Interact:* Fast `lr` + Very Short `gamma` | `0.9` | `0.0005` | `--gamma 0.9 --lr 0.0005` |

---

### 👨‍🔬 Member 4: "Fine-Tuning" Specialist

**Reasoning:** This member's mission is to **"zoom in"** on the most promising parameters from the other groups. Assuming the baseline `lr` and `epsilon_decay` are *close* to correct, this member will test a "grid" of values *around* them to find the "best of the best" combination. The final "Wildcard" test combines the most extreme "patient" settings from all groups.

| Exp | Goal | `lr` | `epsilon_decay` | Command to Run (`--steps 1000000`)... |
| :--- | :--- | :--- | :--- | :--- |
| **4.1** | **Baseline** | `0.0001` | `250000` | ` ` (default) |
| **4.2** | Grid 1: Fast `lr` + Fast Decay | `0.00025` | `150000` | `--lr 0.00025 --epsilon_decay 150000` |
| **4.3** | Grid 2: Fast `lr` + Base Decay | `0.00025` | `250000` | `--lr 0.00025` |
| **4.4** | Grid 3: Fast `lr` + Slow Decay | `0.00025` | `400000` | `--lr 0.00025 --epsilon_decay 400000` |
| **4.5** | Grid 4: Base `lr` + Fast Decay | `0.0001` | `150000` | `--epsilon_decay 150000` |
| **4.6** | Grid 5: Base `lr` + Slow Decay | `0.0001` | `400000` | `--epsilon_decay 400000` |
| **4.7** | Grid 6: Slow `lr` + Fast Decay | `0.00005` | `150000` | `--lr 0.00005 --epsilon_decay 150000` |
| **4.8** | Grid 7: Slow `lr` + Base Decay | `0.00005` | `250000` | `--lr 0.00005` |
| **4.9** | Grid 8: Slow `lr` + Slow Decay | `0.00005` | `400000` | `--lr 0.00005 --epsilon_decay 400000` |
| **4.10** | **Wildcard:** "The Patient Agent" | `0.00005` | `750000` | `--lr 0.00005 --epsilon_decay 750000 --gamma 0.995` |