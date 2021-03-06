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

* Hack [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) package.

    * OSX or Linux
    
        ``` shell
        cp hack/smb_env.py $(dirname $(python -c "import gym_super_mario_bros; print(gym_super_mario_bros.__file__)"))
        ```
    
    * Windows
    
        ``` shell
        python -c "import gym_super_mario_bros; import os; print(os.path.dirname(gym_super_mario_bros.__file__))"
        ```
        
        Copy ``hack/smb_env.py`` to the folder which command above prints out.

## How To Run

```shell
python eval.py --render=1
```
