---
title: "Prompt flowをCLIから使ってみる"
emoji: "🪄"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Azure", "Promptflow", "ChatGPT"]
published: false
---

# はじめに

# ハンズオン

## フローのカスタマイズ

```zsh
% pf connection create --file ../../connections/azure_openai.yml
Connection create failed with StoreConnectionEncryptionKeyError: System keyring backend service not found in your operating system. See https://pypi.org/project/keyring/ to install requirement for different operating system, or 'pip install keyrings.alt' to use the third-party backend. Reach more detail about this error at https://microsoft.github.io/promptflow/how-to-guides/faq.html#connection-creation-failed-with-storeconnectionencryptionkeyerror
```

`pip install keyrings.alt`

```zsh
  pf connection create --file ../../connections/azure_openai.yml
{
    "name": "open_ai_connection",
    "module": "promptflow.connections",
    "created_date": "2024-01-01T17:04:38.601690",
    "last_modified_date": "2024-01-01T17:04:38.601690",
    "type": "azure_open_ai",
    "api_key": "******",
    "api_base": "https://openai-lab-swedencentral.openai.azure.com/",
    "api_type": "azure",
    "api_version": "2023-07-01-preview"
}
```

もともと：

```:chat.jinja2
system:
You are a helpful assistant.

{% for item in chat_history %}
user:
{{item.inputs.question}}
assistant:
{{item.outputs.answer}}
{% endfor %}

user:
{{question}}
```

書き換え後：

```:chat.jinja2
system:
You are an assistant to calculate the answer to the provided math problems.
Please return the final numerical answer only, without any accompanying reasoning or explanation.

{% for item in chat_history %}
user:
{{item.inputs.question}}
assistant:
{{item.outputs.answer}}
{% endfor %}

user:
{{question}}
```

```:flow.dag.yaml
$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json
inputs:
  chat_history:
    type: list
    default: []
  question:
    type: string
    is_chat_input: true
    default: What is ChatGPT?
outputs:
  answer:
    type: string
    reference: ${chat.output}
    is_chat_output: true
nodes:
- inputs:
    # This is to easily switch between openai and azure openai.
    # deployment_name is required by azure openai, model is required by openai.
    deployment_name: gpt-35-turbo
    model: gpt-3.5-turbo
    max_tokens: "256"
    temperature: "0.7"
    chat_history: ${inputs.chat_history}
    question: ${inputs.question}
  name: chat
  type: llm
  source:
    type: code
    path: chat.jinja2
  api: chat
  connection: open_ai_connection
node_variants: {}
environment:
    python_requirements_txt: requirements.txt
```

```zsh
  pf flow test --flow ./basic-chat --inputs question="1+1=?"
2024-01-01 17:06:04 +0900  265398 execution.flow     INFO     Start executing nodes in thread pool mode.
2024-01-01 17:06:04 +0900  265398 execution.flow     INFO     Start to run 1 nodes with concurrency level 16.
2024-01-01 17:06:04 +0900  265398 execution.flow     INFO     Executing node chat. node run id: 6a2241c5-c90e-4cb4-88c5-ee5a1753ae21_chat_0
2024-01-01 17:06:05 +0900  265398 execution.flow     INFO     Node chat completes.
{
    "answer": "2"
}
```

## プロンプトのクオリティを評価

GitHub に載っているサンプルだと実行に失敗したため、ファイルパスなどを若干修正し、下記で実行しました。

```zsh
  pf run create --flow ./basic-chat --data ./chat-math-variant/data.jsonl --column-mapping question='${data.question}' chat_history=\[] --connections chat.connection=open_ai_connection chat.deployment_name=gpt-4-turbo --stream --name $base_run_name
```

実行結果：

```zsh
======= Run Summary =======

Run name: "base_run_2"
Run status: "Completed"
Start time: "2024-01-01 17:23:43.625560"
Duration: "0:00:08.351035"
Output path: "/home/shohei/.promptflow/.runs/base_run_2"



{
    "name": "base_run_2",
    "created_on": "2024-01-01T17:23:43.625560",
    "status": "Completed",
    "display_name": "base_run_2",
    "description": null,
    "tags": null,
    "properties": {
        "flow_path": "/home/shohei/projects/microsoft/promptflow/examples/flows/chat/basic-chat",
        "output_path": "/home/shohei/.promptflow/.runs/base_run_2",
        "system_metrics": {
            "total_tokens": 1854,
            "prompt_tokens": 1788,
            "completion_tokens": 66,
            "duration": 7.278764
        }
    },
    "flow_name": "basic-chat",
    "data": "/home/shohei/projects/microsoft/promptflow/examples/flows/chat/chat-math-variant/data.jsonl",
    "output": "/home/shohei/.promptflow/.runs/base_run_2/flow_outputs"
}
```

実行結果の確認

```zsh
pf run show-details --name $base_run_name
+----+---------------+-----------------+---------------+---------------+
|    | inputs.chat   | inputs.question |   inputs.line | outputs.ans   |
|    | _history      |                 |       _number | wer           |
+====+===============+=================+===============+===============+
|  0 | []            | Compute $\dbi   |             0 | 4368          |
|    |               | nom{16}{5}$.    |               |               |
+----+---------------+-----------------+---------------+---------------+
|  1 | []            | Determine the   |             1 | 60            |
|    |               | number of       |               |               |
|    |               | ways to         |               |               |
|    |               | arrange the     |               |               |
|    |               | letters of      |               |               |
|    |               | the word        |               |               |
|    |               | PROOF.          |               |               |
+----+---------------+-----------------+---------------+---------------+
| ..(省略) | ...           | ...             |...            | ...           |
```

## 実行結果の評価

```zsh
eval_run_name="eval_run"
pf run create --flow ./eval-chat-math --data ../chat/chat-math-variant/data.jsonl --column-mapping groundtruth='${data.answer}' prediction='${run.outputs.answer}' --stream --run $base_run_name --name $eval_run_name
```

実行結果：

```zsh
======= Run Summary =======

Run name: "eval_run"
Run status: "Completed"
Start time: "2024-01-01 17:27:29.090182"
Duration: "0:00:02.342412"
Output path: "/home/shohei/.promptflow/.runs/eval_run"



{
    "name": "eval_run",
    "created_on": "2024-01-01T17:27:29.090182",
    "status": "Completed",
    "display_name": "eval_run",
    "description": null,
    "tags": null,
    "properties": {
        "flow_path": "/home/shohei/projects/microsoft/promptflow/examples/flows/evaluation/eval-chat-math",
        "output_path": "/home/shohei/.promptflow/.runs/eval_run",
        "system_metrics": {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "duration": 1.491181
        }
    },
    "flow_name": "eval-chat-math",
    "data": "/home/shohei/projects/microsoft/promptflow/examples/flows/chat/chat-math-variant/data.jsonl",
    "output": "/home/shohei/.promptflow/.runs/eval_run/flow_outputs",
    "run": "base_run_2"
}
```

評価フローのメトリックを取得

```zsh
  pf run show-metrics --name $eval_run_name
{
    "accuracy": 0.4,
    "error_rate": 0.6
}
```

ブラウザで、`base_run`と`eval_run`の実行結果を一対一で可視化して比較できます。

```zsh
pf run visualize --name "$base_run_name,$eval_run_name"
The HTML file is generated at '/tmp/pf-visualize-detail-dk5ldej6.html'.
Trying to view the result in a web browser...
Failed to visualize from the web browser, the HTML file locates at '/tmp/pf-visualize-detail-dk5ldej6.html'.
You can manually open it with your web browser, or try SDK to visualize it.
```

私が WSL 環境で実行しているためかブラウザの起動自体には失敗していますが、HTML ファイル (`/tmp/pf-visualize-detail-dk5ldej6.html`)は無事に生成されています。こちらをブラウザで表示します。

実行結果の可視化がされています。今回はフローへの 20 個の入力/出力があった中で、正解 (ground truth)と予測値 (チャットフロー実行結果)がそれぞれ一致していたかどうかの結果がテーブルで表示されています。また、右上にはおなじみのグラフビューでフローを構成するノードが確認できます。

![](/images/flow-viz-1.png)

## プロンプトの改善とその評価

プロンプトの改善をしていきます。Chain-of-Thought (CoT)のアプローチを取り入れたプロンプトのフローを実行し、その回答精度を評価します。
`/chat/chat-math-variant`フォルダに 2 つの新たなプロンプトが用意されています。

```:chat_variant_1.jinja2
system:
You are an assistant to calculate the answer to the provided math problems.
Please think step by step.
Return the final numerical answer only and any accompanying reasoning or explanation seperately as json format.

user:
A jar contains two red marbles, three green marbles, ten white marbles and no other marbles. Two marbles are randomly drawn from this jar without replacement. What is the probability that these two marbles drawn will both be red? Express your answer as a common fraction.
assistant:
{Chain of thought: "The total number of marbles is $2+3+10=15$.  The probability that the first marble drawn will be red is $2/15$.  Then, there will be one red left, out of 14.  Therefore, the probability of drawing out two red marbles will be: $$\\frac{2}{15}\\cdot\\frac{1}{14}=\\boxed{\\frac{1}{105}}$$.", "answer": "1/105"}
user:
Find the greatest common divisor of $7!$ and $(5!)^2.$
assistant:
{"Chain of thought": "$$ \\begin{array} 7! &=& 7 \\cdot 6 \\cdot 5 \\cdot 4 \\cdot 3 \\cdot 2 \\cdot 1 &=& 2^4 \\cdot 3^2 \\cdot 5^1 \\cdot 7^1 \\\\ (5!)^2 &=& (5 \\cdot 4 \\cdot 3 \\cdot 2 \\cdot 1)^2 &=& 2^6 \\cdot 3^2 \\cdot 5^2 \\\\ \\text{gcd}(7!, (5!)^2) &=& 2^4 \\cdot 3^2 \\cdot 5^1 &=& \\boxed{720} \\end{array} $$.", "answer": "720"}
{% for item in chat_history %}

user:
{{item.inputs.question}}
assistant:
{{item.outputs.answer}}
{% endfor %}

user:
{{question}}
```

```:chat_variant_2.jinja2
system:
You are an assistant to calculate the answer to the provided math problems.
Please think step by step.
Return the final numerical answer only and any accompanying reasoning or explanation seperately as json format.

user:
A jar contains two red marbles, three green marbles, ten white marbles and no other marbles. Two marbles are randomly drawn from this jar without replacement. What is the probability that these two marbles drawn will both be red? Express your answer as a common fraction.
assistant:
{Chain of thought: "The total number of marbles is $2+3+10=15$.  The probability that the first marble drawn will be red is $2/15$.  Then, there will be one red left, out of 14.  Therefore, the probability of drawing out two red marbles will be: $$\\frac{2}{15}\\cdot\\frac{1}{14}=\\boxed{\\frac{1}{105}}$$.", "answer": "1/105"}
user:
Find the greatest common divisor of $7!$ and $(5!)^2.$
assistant:
{"Chain of thought": "$$ \\begin{array} 7! &=& 7 \\cdot 6 \\cdot 5 \\cdot 4 \\cdot 3 \\cdot 2 \\cdot 1 &=& 2^4 \\cdot 3^2 \\cdot 5^1 \\cdot 7^1 \\\\ (5!)^2 &=& (5 \\cdot 4 \\cdot 3 \\cdot 2 \\cdot 1)^2 &=& 2^6 \\cdot 3^2 \\cdot 5^2 \\\\ \\text{gcd}(7!, (5!)^2) &=& 2^4 \\cdot 3^2 \\cdot 5^1 &=& \\boxed{720} \\end{array} $$.", "answer": "720"}
user:
A club has 10 members, 5 boys and 5 girls.  Two of the members are chosen at random.  What is the probability that they are both girls?
assistant:
{"Chain of thought": "There are $\\binomial{10}{2} = 45$ ways to choose two members of the group, and there are $\\binomial{5}{2} = 10$ ways to choose two girls.  Therefore, the probability that two members chosen at random are girls is $\\dfrac{10}{45} = \\boxed{\\dfrac{2}{9}}$.", "answer": "2/9"}
user:
Allison, Brian and Noah each have a 6-sided cube. All of the faces on Allison's cube have a 5. The faces on Brian's cube are numbered 1, 2, 3, 4, 5 and 6. Three of the faces on Noah's cube have a 2 and three of the faces have a 6. All three cubes are rolled. What is the probability that Allison's roll is greater than each of Brian's and Noah's? Express your answer as a common fraction.
assistant:
{"Chain of thought": "Since Allison will always roll a 5, we must calculate the probability that both Brian and Noah roll a 4 or lower. The probability of Brian rolling a 4 or lower is $\\frac{4}{6} = \\frac{2}{3}$ since Brian has a standard die. Noah, however, has a $\\frac{3}{6} = \\frac{1}{2}$ probability of rolling a 4 or lower, since the only way he can do so is by rolling one of his 3 sides that have a 2. So, the probability of both of these independent events occurring is $\\frac{2}{3} \\cdot \\frac{1}{2} = \\boxed{\\frac{1}{3}}$.", "answer": "1/3"}
user:
Compute $\\density binomial{50}{2}$.
assistant:
{"Chain of thought": "$\\density binomial{50}{2} = \\dfrac{50!}{2!48!}=\\dfrac{50\\times 49}{2\\times 1}=\\boxed{1225}.$", "answer": "1225"}
user:
The set $S = \\{1, 2, 3, \\ldots , 49, 50\\}$ contains the first $50$ positive integers.  After the multiples of 2 and the multiples of 3 are removed, how many integers remain in the set $S$?
assistant:
{"Chain of thought": "The set $S$ contains $25$ multiples of 2 (that is, even numbers).  When these are removed, the set $S$ is left with only the odd integers from 1 to 49. At this point, there are $50-25=25$ integers in $S$. We still need to remove the multiples of 3 from $S$.\n\nSince $S$ only contains odd integers after the multiples of 2 are removed,  we must remove the odd multiples of 3 between 1 and 49.  These are 3, 9, 15, 21, 27, 33, 39, 45, of which there are 8.  Therefore, the number of integers remaining in the set $S$ is $25 - 8 = \\boxed{17}$.", "answer": "17"}
{% for item in chat_history %}

user:
{{item.inputs.question}}
assistant:
{{item.outputs.answer}}
{% endfor %}

user:
{{question}}
```

どちらのプロンプトもステップバイステップで考えよう`Please think step by step.`という記載と、その後に実際に段階的に考えさせている few-shot 例が記載されています。`chat_variant_2.jinja2`の方がより例を多く記載しているので、これまで利用してきた `chat.jinja2`のプロンプトと合わせて、３つのプロンプトの実行結果を比較していきましょう。

## プロンプトバリアントのテストと評価

`promptflow/examples/flows`フォルダに移動し、フローの実行名を環境変数に設定します。

```zsh
cd ..
base_run_name="base_run_variant_"
eval_run_name="eval_run_variant_"
```

フローを実行します。もし Azure OpenAI とのコネクション名をデフォルトから変更している場合は、それに合わせて`flow.dag.yaml`の`connection`パラメーターを変更してください。

```zsh
# Test and evaluate variant_0:
# Test-run
pf run create --flow ./chat/chat-math-variant --data ./chat/chat-math-variant/data.jsonl --column-mapping question='${data.question}' chat_history=\[] --variant '${chat.variant_0}' --stream  --name "${base_run_name}0"
# Evaluate-run
pf run create --flow ./evaluation/eval-chat-math --data ./chat/chat-math-variant/data.jsonl --column-mapping groundtruth='${data.answer}' prediction='${run.outputs.answer}' --stream --run "${base_run_name}0" --name "${eval_run_name}0"

# Test and evaluate variant_1:
# Test-run
pf run create --flow ./chat/chat-math-variant --data ./chat/chat-math-variant/data.jsonl --column-mapping question='${data.question}' chat_history=\[] --variant '${chat.variant_1}' --stream --name "${base_run_name}1"
# Evaluate-run
pf run create --flow ./evaluation/eval-chat-math --data ./chat/chat-math-variant/data.jsonl --column-mapping groundtruth='${data.answer}' prediction='${run.outputs.answer}' --stream --run "${base_run_name}1" --name "${eval_run_name}1"

# Test and evaluate variant_2:
# Test-run
pf run create --flow ./chat/chat-math-variant --data ./chat/chat-math-variant/data.jsonl --column-mapping question='${data.question}' chat_history=\[] --variant '${chat.variant_2}' --stream --name "${base_run_name}2"
# Evaluate-run
pf run create --flow ./evaluation/eval-chat-math --data ./chat/chat-math-variant/data.jsonl --column-mapping groundtruth='${data.answer}' prediction='${run.outputs.answer}' --stream --run "${base_run_name}2" --name "${eval_run_name}2"
```

評価フローのメトリックを取得します。

```zsh
pf run show-metrics --name "${eval_run_name}0"
pf run show-metrics --name "${eval_run_name}1"
pf run show-metrics --name "${eval_run_name}2"
```

実行結果：

```json
{
    "accuracy": 0.45,
    "error_rate": 0.55
}
{
    "accuracy": 0.8,
    "error_rate": 0.2
}
{
    "accuracy": 0.95,
    "error_rate": 0.05
}
```

※LLM 出力にはランダム性があるため、もとのサンプルコードと私の実行結果は異なっています。

結果を可視化します。

```zsh
pf run visualize --name "${base_run_name}0,${eval_run_name}0,${base_run_name}1,${eval_run_name}1,${base_run_name}2,${eval_run_name}2"
```

出力された HTML ファイルをブラウザで開きます。

![](/images/flow-viz-2.png)

左上の Runs & Metrics パネル内の`evel_run_variant_2`が一番 accuracy と error rate の値が良いのが分かります。つまり`chat_variant_2.jinja2`が一番よい回答を生成したということです。

ただし、variant_2 の accuracy が 0.95 なのに対して、variant_1 の accuracy は 0.8 と比較的高い値になっています。variant_2 は variant_1 よりもかなり入力トークン数が多く、実行コストがかかるプロンプトです。実際のビジネス上問題がなければコストパフォーマンスのよい variant_1 のプロンプト利用も選択肢に入るでしょう。

なお、20 個あるテストデータの各データに対しての実行結果は、Outputs パネル内で確認できます。

# おわりに