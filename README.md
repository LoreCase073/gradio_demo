# Large-Vision Models research

## Description

This project is aimed to research and implement event detection and automatic description in videos using new Large-Vision Models (LVM) architectures. In particular, we focus on the use of Llava-Next-Video to create a functional chatbot demo to analyze video and interacting with the LVM. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### FFMPEG

To install FFMPEG via terminal, you can use the following command:

```bash
sudo apt-get install ffmpeg
```

This command will install FFMPEG on your system. Make sure you have administrative privileges to run this command.

### Conda Environment

To install the Conda environment using the requirements.txt file, you can use the following command:

```bash
conda create --name myenv python=3.11.9
```

This command will create a new Conda environment called "myenv".
To install the requirements, activate the conda environment:

```bash
conda activate myenv
```

Clone the repository:

```bash
git clone https://github.com/LoreCase073/gradio_demo.git
```

and install all the packages listed in the requirements.txt file by running:

```
pip install -r requirements.txt
```


## Usage

After creating the Conda environment as specified above, you can start the demo using the following command:

```
python gradio_llava.py
```

To instantiate a demo with authentication and public url:

```
python gradio_llava.py --user id_user --password password_for_authentication
```
where *id_user* and *password_for_authentication* are the credentials to login into the demo.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
