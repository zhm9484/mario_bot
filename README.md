# mario-bot

A bot plays [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) using PyTorch model.

## Installation

* Python >= 3.6

* Install requirements.

    ```shell
    pip install -r requirements.txt
    ```

* Install PyTorch.

    * OSX

        ```shell
        pip install torch==1.2.0
        ```
    
    * Linux or Windows

        ```shell
        pip install torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
        ```

## How To Run

```shell
python eval.py --render=1
```
