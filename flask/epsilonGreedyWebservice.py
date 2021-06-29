from flask import Flask, jsonify, render_template, request
import time
from epsilonGreedyLib import Decision
import numpy as np
import matplotlib.pyplot as plt
import logging


app = Flask(__name__)

@app.route("/")
def main():
    return render_template('main.html', reload = time.time())

@app.route("/api/info")
def api_info():
    info = {
       "ip" : "127.0.0.1",
       "hostname" : "everest",
       "description" : "Main server",
       "load" : [ 3.21, 7, 14 ]
    }
    return jsonify(info)

@app.route("/api/calc")
def add():
    a = int(request.args.get('a', 0))
    b = int(request.args.get('b', 0))
    div = 'na'
    if b != 0:
        div = a/b
    return jsonify({
        "a"        :  a,
        "b"        :  b,
        "add"      :  a+b,
        "multiply" :  a*b,
        "subtract" :  a-b,
        "divide"   :  div,
    })

@app.route("/api/reward")
def reward():
    true_rewards = [np.random.randn() for _ in range(10)]
    decisions = [Decision(true_reward) for true_reward in true_rewards]
    print(decisions)
    return jsonify({
        "user_reward"        :  "0.46",
        "computer_reward"        :  "0.9"
    })
