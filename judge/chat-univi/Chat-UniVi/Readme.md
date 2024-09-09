### Setup Instructions

To run the `Chat-UniVi` project, follow these steps to install the necessary libraries.
1. Clone the `Chat-UniVi` repository:
   ```bash
   git clone https://github.com/PKU-YuanGroup/Chat-UniVi
   ```

2. Navigate to the project directory:
   ```bash
   cd Chat-UniVi
   ```

3. Upgrade pip and install project dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

### Optional Steps
- Create and activate a conda environment:
   ```bash
   conda create -n chatunivi python=3.10 -y
   conda activate chatunivi
   ```
- If you only intend to perform inference, you do not need to install `ninja` and `flash-attn`:
   ```bash
   # pip install ninja  
   # pip install flash-attn --no-build-isolation
   ```

By following these steps, you can successfully run the `Chat-UniVi` project and perform inference or development tasks.
