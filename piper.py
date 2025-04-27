import subprocess


class PiperTTS:
    def synthesize(self, text, model_path):
        process = subprocess.Popen(
            ["piper", "--model", model_path, "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout_data, stderr_data = process.communicate(input=text.encode())

        if process.returncode != 0:
            raise RuntimeError(f"Piper failed: {stderr_data.decode()}")

        return stdout_data  # these are your raw audio bytes
