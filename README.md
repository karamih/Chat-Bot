# Chat-Bot
*The simplest Chat Bot in Persian with Pytorch.*
<hr>
<div>
  <h2> How does output looks like?</h2>
  <img src="result.png">
  <p>In this case we just use the most basic methods for bulding a chat bot.<br><br>embedding: one_hot <br>model: mlp<br><br>
  There is a `data.json` file, it contains data which we train on. for changing the capabalities of bot you can edit this file and then running `train.py` file. But before train on your data, you probably need to change the model structure in `model.py` file.<br><br>
    So for fine tunning this chat bot on your data, follow these steps:<br><br>
    1) modify the `data.json` file.<br><br>
    2) modify the model structure in `model.py file.`<br><br>
    3) run `train.py` file.<br><br>
    4) run `main.py` file and chat with bot.</p>
</div><hr>
<em>You can run this project in pycharm to see chats in Persian.</em>
