# Run Google Colab on Your Local GPU via VS Code

This guide shows how to use Google Colab's UI inside your browser (or VS Code Simple Browser), but run code on your local machine (with your own GPU, RAM, and files). This is ideal for advanced workflows, debugging, or using your own hardware for Colab notebooks.

---

## 1. Install Required Packages (PowerShell in VS Code)

```powershell
python -m pip install jupyter_http_over_ws notebook
python -m jupyter serverextension enable --py jupyter_http_over_ws
```

## 2. Start Jupyter Notebook Server for Colab

```powershell
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.token='YOUR_TOKEN' --NotebookApp.disable_check_xsrf=True
```
- Replace `YOUR_TOKEN` with a secure string (e.g., `mysecret123`).
- Keep this PowerShell terminal open while using Colab.

## 3. Connect Colab to Local Runtime

1. Open [Google Colab](https://colab.research.google.com) in your browser or VS Code Simple Browser.
2. Go to: `Connect` (top right) → `Connect to local runtime`.
3. Enter:
   ```
   http://localhost:8888/?token=YOUR_TOKEN
   ```
   (Use the same token as above.)
4. Click `Connect`.

## 4. Run Your Notebook
- You can now run any Colab notebook, and it will execute on your local machine (using your local GPU, files, and Python environment).
- You can access local files, install packages, and use your own hardware.

---

## Troubleshooting
- **Colab cannot connect:** Ensure the Jupyter server is running and the port/token match.
- **Firewall issues:** Allow port 8888 in Windows Firewall if needed.
- **Authentication:** Never use `--NotebookApp.token=''` on public networks. Always set a token for security.
- **GPU not detected:** Ensure your local Python environment has CUDA, PyTorch/TensorFlow, and drivers installed.
- **VS Code Simple Browser:** Some Google auth flows may require using your default browser.

---

## Tips
- You can edit notebooks in VS Code, then run them in Colab UI (local runtime).
- For advanced workflows, use VS Code's Python extension for debugging and Colab for interactive runs.
- To stop the server, close the PowerShell terminal running Jupyter.

---

## References
- [Colab Local Runtime Docs](https://research.google.com/colaboratory/local-runtimes.html)
- [Jupyter HTTP over WebSockets](https://github.com/googlecolab/jupyter_http_over_ws)

---

Feel free to copy/paste these commands and steps. For more help, ask in this repo or open an issue!
6