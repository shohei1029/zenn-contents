---
title: "Prompt flowをCLIから使ってみる"
emoji: "🪄"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Azure", "Promptflow", "ChatGPT"]
published: false
publication_name: microsoft
---

# はじめに

[Prompt flow](https://learn.microsoft.com/ja-jp/azure/machine-learning/prompt-flow/overview-what-is-prompt-flow) (プロンプトフローとも表記)は Microsoft が開発している開発者用ツールで、LLM を組み込んだアプリケーションのためのプロンプト構築・評価、そしてデプロイをサポートします。

Azure Machine Learning から簡単に利用でき、分かりやすい解説記事も登場しています。

- [Prompt Flow が使えるようになったから、もう LangChain とか自分でホストしなくていい世界になったのかもしれない。 | DevelopersIO](https://dev.classmethod.jp/articles/azureml-prompt-flow-ktkr/)
- [【日本最速？】Azure AI Studio で Prompt Flow を触ってみた感想 #Azure - Qiita](https://qiita.com/lazy-kz/items/5f6f8dc821d25fc484db)

Prompt flow はもともとは Azure Machine Learning 内の機能だったのですが、いまは**Azure Machine Learning とは独立して使用できるツール**です。**CLI/SDK や VS Code 拡張機能が提供**されており、それらの中で完結して利用できます。

ただ、Prompt flow は Azure Machine Learning の中で使うツールというイメージが広がっているため、本記事では Prompt flow の CLI を使ってフローの作成、評価、チューニングを行う流れを紹介します。

リポジトリ：https://github.com/microsoft/promptflow

# 手順

## 前提条件

Azure OpenAI のリソースと、`gpt-35-turbo`モデルのデプロイが完了している前提です。ただ、後述するように接続設定のファイルを変更すれば OpenAI API でも利用できます。

## Prompt flow のインストール

`pip`でインストールします。

```shell
pip install promptflow promptflow-tools
```

執筆時点では Python 3.9 系が推奨されています。

## リポジトリの準備

本記事は基本的にこちらの[チュートリアル](https://github.com/microsoft/promptflow/blob/main/examples/tutorials/flow-fine-tuning-evaluation/promptflow-quality-improvement.md)の内容に沿って実施しています。

チュートリアルが含まれるリポジトリを clone します。

```shell
git clone https://github.com/microsoft/promptflow.git
```

```shell
cd promptflow/examples/tutorials/flow-fine-tuning-evaluation
```

## Azure OpenAI (または OpenAI API)との接続設定

`azure_openai.yml`の内容をもとに Azure OpenAI Service との接続構成を設定します。
ファイルの中身はこのような YAML ファイルになっており、Azure portal でデプロイした Azure OpenAI リソースのキーや API base の情報を記載します。Azure OpenAI ではなく OpenAI 社の API を使うことも可能です。

```yaml:azure_openai.yml
$schema: https://azuremlschemas.azureedge.net/promptflow/latest/AzureOpenAIConnection.schema.json
name: open_ai_connection
type: azure_open_ai
api_key: "<Your_API_key>"
api_base: "https://openai-lab-swedencentral.openai.azure.com/"
api_type: "azure"
```

:::message
OpenAI API を使う場合はこのような書き方になります。このファイルも同じディレクトに置かれています。

```yaml:openai.yml
$schema: https://azuremlschemas.azureedge.net/promptflow/latest/OpenAIConnection.schema.json
name: open_ai_connection
type: open_ai
api_key: "<user-input>"
organization: "" # optional
```

:::

次のコマンドでコネクションを作成します。

```shell
pf connection create --file ../../connections/azure_openai.yml
```

:::message
`azure_openai.yml`ファイルを書き換えなくても、`pf connection create`コマンド実行時の引数で各項目を上書きできます。

```shell
# Override keys with --set to avoid yaml file changes
pf connection create --file ../../connections/azure_openai.yml --set api_key=<your_api_key> api_base=<your_api_base> --name open_ai_connection
```

:::

:::message
私が`pf connection create`を実行したところ、`Connection create failed with StoreConnectionEncryptionKeyError`というエラーが発生しました (WSL2, Ubuntu 22.04 環境)。どうやら WSL 環境で発生するエラーのようです ([FAQ](https://microsoft.github.io/promptflow/how-to-guides/faq.html#connection-creation-failed-with-storeconnectionencryptionkeyerror)より)

```shell
Connection create failed with StoreConnectionEncryptionKeyError: System keyring backend service not found in your operating system. See https://pypi.org/project/keyring/ to install requirement for different operating system, or 'pip install keyrings.alt' to use the third-party backend. Reach more detail about this error at https://microsoft.github.io/promptflow/how-to-guides/faq.html#connection-creation-failed-with-storeconnectionencryptionkeyerror
```

上記エラーが発生した場合は次のコマンドで`keyrings.alt`をインストールしてください。

```shell
pip install keyrings.alt
```

:::

`pf connection create`の実行がうまくいくと、次のような結果が表示されます。

```json
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

## フローのカスタマイズ

もともと用意されているチャットフローを修正していきます。
まず、フローが格納されているディレクトリへ移動します。

```shell
cd ../../flows/chat/basic-chat/
```

`chat.jinja2`の中身を書き換えます。

書き換え前：

```jinja2:chat.jinja2
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

```jinja2:chat.jinja2
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

フロー全体の構成設定は`flow.dag.yaml`に記載されています。ここで Azure OpenAI や他に利用する外部サービスとの接続設定 (モデル名、temperature などのパラメーター設定を含む)、フローの入出力、処理を行うノードの定義などを行います。

```yaml:flow.dag.yaml
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

:::message
今回は Azure OpenAI`gpt-35-turbo`のモデル (デプロイメント)になっていますが、デプロイ名が異なる場合は修正します。また、connection 名を上記の`azure_openai.yml`とは違う名前に変更している場合はこちらも合わせて変更してください。
:::

## フローのテスト実行

1 つ上のディレクトリへ行き、先ほどのファイル群がある`basic-chat`ディレクトリをプロンプトのフローとして実行します。フローへの入力として`question="1+1=?"`を与えています。

```shell
cd ..
pf flow test --flow ./basic-chat --inputs question="1+1=?"
```

実行ログが表示された後に、JSON 形式で結果が出力されます。

```shell
2024-01-01 17:06:04 +0900  265398 execution.flow     INFO     Start executing nodes in thread pool mode.
2024-01-01 17:06:04 +0900  265398 execution.flow     INFO     Start to run 1 nodes with concurrency level 16.
2024-01-01 17:06:04 +0900  265398 execution.flow     INFO     Executing node chat. node run id: 6a2241c5-c90e-4cb4-88c5-ee5a1753ae21_chat_0
2024-01-01 17:06:05 +0900  265398 execution.flow     INFO     Node chat completes.
{
    "answer": "2"
}
```

入力`question="1+1=?"`に対する出力として正しく`"2"`と正解しています。

もっと複雑な入力を与えてみましょう。

```shell
pf flow test --flow ./basic-chat --inputs question="We are allowed to remove exactly one integer from the list $$-1,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,$$and then we choose two distinct integers at random from the remaining list. What number should we remove if we wish to maximize the probability that the sum of the two chosen numbers is 10?"
```

出力：

```JSON
{
    "answer": "-1"
}
```

正しい計算結果は 5 なので、誤った出力が行われていることが分かります。ここから、プロンプトを修正してもっと複雑な数値問題にもうまく回答できるようにしていきます。

## プロンプトのクオリティを評価

Prompt flow では、複数の入力をまとめたファイルを用意してフローをバッチ実行し、それぞれの結果を評価できます。

`promptflow/examples/flows/chat/chat-math-variant/data.jsonl`には 20 個のテストデータが格納されています。
中身は、入力となる質問 (`question`)、正解の値 (`answer`)、フローの実行結果 (`raw_answer`)から構成されています。

```JSON
{
    "question": "Determine the number of ways to arrange the letters of the word PROOF.",
    "answer": "60",
    "raw_answer": "There are two O's and five total letters, so the answer is $\\dfrac{5!}{2!} = \\boxed{60}$."
}
```

まず、このデータの question 部分を入力とし、ここまでで利用した`basic-chat`のフローに対してバッチ実行してみましょう。

```shell
base_run_name="base_run"
pf run create --flow ./basic-chat --data ./chat-math-variant/data.jsonl --column-mapping question='${data.question}' chat_history=\[] --connections chat.connection=open_ai_connection chat.deployment_name=gpt-35-turbo --stream --name $base_run_name
```

なお、コピペで実行しやすいよう`chat.deployment_name=gpt-35-turbo`としていますが、私は GPT-4 Turbo を利用したため、実際は`chat.deployment_name=gpt-4-turbo`で実行しています。
また、リポジトリに載っているサンプルそのままだと実行に失敗したため、ファイルパスなどは若干修正しました。

実行結果としてテキストの情報と JSON が出力されます。ここで消費したトークン数も確認できます。

```shell
======= Run Summary =======

Run name: "base_run"
Run status: "Completed"
Start time: "2024-01-01 17:23:43.625560"
Duration: "0:00:08.351035"
Output path: "/home/shohei/.promptflow/.runs/base_run"



{
    "name": "base_run_",
    "created_on": "2024-01-01T17:23:43.625560",
    "status": "Completed",
    "display_name": "base_run",
    "description": null,
    "tags": null,
    "properties": {
        "flow_path": "/home/shohei/projects/microsoft/promptflow/examples/flows/chat/basic-chat",
        "output_path": "/home/shohei/.promptflow/.runs/base_run",
        "system_metrics": {
            "total_tokens": 1854,
            "prompt_tokens": 1788,
            "completion_tokens": 66,
            "duration": 7.278764
        }
    },
    "flow_name": "basic-chat",
    "data": "/home/shohei/projects/microsoft/promptflow/examples/flows/chat/chat-math-variant/data.jsonl",
    "output": "/home/shohei/.promptflow/.runs/base_run/flow_outputs"
}
```

`pf run show-details`コマンドを利用することで実行結果を各データごとに確認できます。`inputs.question`がフローへの入力で、`outputs.answer`がフローの出力です。

```shell
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

:::details 省略していない結果はこちら

```shell
  pf run show-details --name $base_run_name
+----+----------------+--------------+--------------+----------------+
|    | inputs.quest   | inputs.cha   |   inputs.lin | outputs.answ   |
|    | ion            | t_history    |     e_number | er             |
+====+================+==============+==============+================+
|  0 | Compute $\db   | []           |            0 | 4368           |
|    | inom{16}{5}$   |              |              |                |
|    | .              |              |              |                |
+----+----------------+--------------+--------------+----------------+
|  1 | Determine      | []           |            1 | 60             |
|    | the number     |              |              |                |
|    | of ways to     |              |              |                |
|    | arrange the    |              |              |                |
|    | letters of     |              |              |                |
|    | the word       |              |              |                |
|    | PROOF.         |              |              |                |
+----+----------------+--------------+--------------+----------------+
|  2 | 23 people      | []           |            2 | 253            |
|    | attend a       |              |              |                |
|    | party. Each    |              |              |                |
|    | person         |              |              |                |
|    | shakes hands   |              |              |                |
|    | with at most   |              |              |                |
|    | 22 other       |              |              |                |
|    | people. What   |              |              |                |
|    | is the         |              |              |                |
|    | maximum        |              |              |                |
|    | possible       |              |              |                |
|    | number of      |              |              |                |
|    | handshakes,    |              |              |                |
|    | assuming       |              |              |                |
|    | that any two   |              |              |                |
|    | people can     |              |              |                |
|    | shake hands    |              |              |                |
|    | at most        |              |              |                |
|    | once?          |              |              |                |
+----+----------------+--------------+--------------+----------------+
|  3 | James has 7    | []           |            3 | 1/7            |
|    | apples. 4 of   |              |              |                |
|    | them are       |              |              |                |
|    | red, and 3     |              |              |                |
|    | of them are    |              |              |                |
|    | green. If he   |              |              |                |
|    | chooses 2      |              |              |                |
|    | apples at      |              |              |                |
|    | random, what   |              |              |                |
|    | is the         |              |              |                |
|    | probability    |              |              |                |
|    | that both      |              |              |                |
|    | the apples     |              |              |                |
|    | he chooses     |              |              |                |
|    | are green?     |              |              |                |
+----+----------------+--------------+--------------+----------------+
|  4 | We are         | []           |            4 | 4              |
|    | allowed to     |              |              |                |
|    | remove         |              |              |                |
|    | exactly one    |              |              |                |
|    | integer from   |              |              |                |
|    | the list       |              |              |                |
|    | $$-1,0, 1,     |              |              |                |
|    | 2, 3, 4, 5,    |              |              |                |
|    | 6, 7, 8, 9,    |              |              |                |
|    | 10,11,$$and    |              |              |                |
|    | then we        |              |              |                |
|    | choose two     |              |              |                |
|    | distinct       |              |              |                |
|    | integers at    |              |              |                |
|    | random from    |              |              |                |
|    | the            |              |              |                |
|    | remaining      |              |              |                |
|    | list.  What    |              |              |                |
|    | number         |              |              |                |
|    | should we      |              |              |                |
|    | remove if we   |              |              |                |
|    | wish to        |              |              |                |
|    | maximize the   |              |              |                |
|    | probability    |              |              |                |
|    | that the sum   |              |              |                |
|    | of the two     |              |              |                |
|    | chosen         |              |              |                |
|    | numbers is     |              |              |                |
|    | 10?            |              |              |                |
+----+----------------+--------------+--------------+----------------+
|  5 | The numbers    | []           |            5 | 2/5            |
|    | 1 through 25   |              |              |                |
|    | are written    |              |              |                |
|    | on 25 cards    |              |              |                |
|    | with one       |              |              |                |
|    | number on      |              |              |                |
|    | each card.     |              |              |                |
|    | Sara picks     |              |              |                |
|    | one of the     |              |              |                |
|    | 25 cards at    |              |              |                |
|    | random. What   |              |              |                |
|    | is the         |              |              |                |
|    | probability    |              |              |                |
|    | that the       |              |              |                |
|    | number on      |              |              |                |
|    | her card       |              |              |                |
|    | will be a      |              |              |                |
|    | multiple of    |              |              |                |
|    | 2 or 5?        |              |              |                |
|    | Express your   |              |              |                |
|    | answer as a    |              |              |                |
|    | common         |              |              |                |
|    | fraction.      |              |              |                |
+----+----------------+--------------+--------------+----------------+
|  6 | A bag has 3    | []           |            6 | 3/16           |
|    | red marbles    |              |              |                |
|    | and 5 white    |              |              |                |
|    | marbles.       |              |              |                |
|    | Two marbles    |              |              |                |
|    | are drawn      |              |              |                |
|    | from the bag   |              |              |                |
|    | and not        |              |              |                |
|    | replaced.      |              |              |                |
|    | What is the    |              |              |                |
|    | probability    |              |              |                |
|    | that the       |              |              |                |
|    | first marble   |              |              |                |
|    | is red and     |              |              |                |
|    | the second     |              |              |                |
|    | marble is      |              |              |                |
|    | white?         |              |              |                |
+----+----------------+--------------+--------------+----------------+
|  7 | Find the       | []           |            7 | The largest    |
|    | largest        |              |              | prime          |
|    | prime          |              |              | divisor of     |
|    | divisor of     |              |              | 11! + 12! is   |
|    | 11! + 12!      |              |              | 13.            |
+----+----------------+--------------+--------------+----------------+
|  8 | These two      | []           |            8 | 2/3            |
|    | spinners are   |              |              |                |
|    | divided into   |              |              |                |
|    | thirds and     |              |              |                |
|    | quarters, re   |              |              |                |
|    | spectively.    |              |              |                |
|    | If each of     |              |              |                |
|    | these          |              |              |                |
|    | spinners is    |              |              |                |
|    | spun once,     |              |              |                |
|    | what is the    |              |              |                |
|    | probability    |              |              |                |
|    | that the       |              |              |                |
|    | product of     |              |              |                |
|    | the results    |              |              |                |
|    | of the two     |              |              |                |
|    | spins will     |              |              |                |
|    | be an even     |              |              |                |
|    | number?        |              |              |                |
|    | Express your   |              |              |                |
|    | answer as a    |              |              |                |
|    | common         |              |              |                |
|    | fraction.      |              |              |                |
|    | [asy]  size(   |              |              |                |
|    | 5cm,5cm);  d   |              |              |                |
|    | raw(Circle((   |              |              |                |
|    | 0,0),1));  d   |              |              |                |
|    | raw(Circle((   |              |              |                |
|    | 3,0),1));  d   |              |              |                |
|    | raw((0,0)--(   |              |              |                |
|    | 0,1));  draw   |              |              |                |
|    | ((0,0)--(-0.   |              |              |                |
|    | 9,-0.47));     |              |              |                |
|    | draw((0,0)--   |              |              |                |
|    | (0.9,-0.47))   |              |              |                |
|    | ;  draw((2,0   |              |              |                |
|    | )--(4,0));     |              |              |                |
|    | draw((3,1)--   |              |              |                |
|    | (3,-1));  la   |              |              |                |
|    | bel("$3$",(-   |              |              |                |
|    | 0.5,0.3));     |              |              |                |
|    | label("$4$",   |              |              |                |
|    | (0.5,0.3));    |              |              |                |
|    | label("$5$",   |              |              |                |
|    | (0,-0.5));     |              |              |                |
|    | label("$5$",   |              |              |                |
|    | (2.6,-0.4));   |              |              |                |
|    | label("$6$",   |              |              |                |
|    | (2.6,0.4));    |              |              |                |
|    | label("$7$",   |              |              |                |
|    | (3.4,0.4));    |              |              |                |
|    | label("$8$",   |              |              |                |
|    | (3.4,-0.4));   |              |              |                |
|    | draw((0,0)--   |              |              |                |
|    | (0.2,0.8),Ar   |              |              |                |
|    | row);  draw(   |              |              |                |
|    | (3,0)--(3.2,   |              |              |                |
|    | 0.8),Arrow);   |              |              |                |
|    | [/asy]         |              |              |                |
+----+----------------+--------------+--------------+----------------+
|  9 | No two         | []           |            9 | 5/26           |
|    | students in    |              |              |                |
|    | Mrs. Vale's    |              |              |                |
|    | 26-student     |              |              |                |
|    | mathematics    |              |              |                |
|    | class have     |              |              |                |
|    | the same two   |              |              |                |
|    | initials.      |              |              |                |
|    | Each           |              |              |                |
|    | student's      |              |              |                |
|    | first name     |              |              |                |
|    | and last       |              |              |                |
|    | name begin     |              |              |                |
|    | with the       |              |              |                |
|    | same letter.   |              |              |                |
|    | If the         |              |              |                |
|    | letter ``Y''   |              |              |                |
|    | is             |              |              |                |
|    | considered a   |              |              |                |
|    | vowel, what    |              |              |                |
|    | is the         |              |              |                |
|    | probability    |              |              |                |
|    | of randomly    |              |              |                |
|    | picking a      |              |              |                |
|    | student        |              |              |                |
|    | whose          |              |              |                |
|    | initials are   |              |              |                |
|    | vowels?        |              |              |                |
|    | Express your   |              |              |                |
|    | answer as a    |              |              |                |
|    | common         |              |              |                |
|    | fraction.      |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 10 | What is the    | []           |           10 | 3.5            |
|    | expected       |              |              |                |
|    | value of the   |              |              |                |
|    | roll of a      |              |              |                |
|    | standard       |              |              |                |
|    | 6-sided die?   |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 11 | How many       | []           |           11 | 8              |
|    | positive       |              |              |                |
|    | divisors of    |              |              |                |
|    | 30! are        |              |              |                |
|    | prime?         |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 12 | Marius is      | []           |           12 | 120 ways.      |
|    | entering a     |              |              |                |
|    | wildlife       |              |              |                |
|    | photo          |              |              |                |
|    | contest, and   |              |              |                |
|    | wishes to      |              |              |                |
|    | arrange his    |              |              |                |
|    | seven snow     |              |              |                |
|    | leopards of    |              |              |                |
|    | different      |              |              |                |
|    | heights in a   |              |              |                |
|    | row. If the    |              |              |                |
|    | shortest two   |              |              |                |
|    | leopards       |              |              |                |
|    | have           |              |              |                |
|    | inferiority    |              |              |                |
|    | complexes      |              |              |                |
|    | and demand     |              |              |                |
|    | to be placed   |              |              |                |
|    | at the ends    |              |              |                |
|    | of the row,    |              |              |                |
|    | how many       |              |              |                |
|    | ways can he    |              |              |                |
|    | line up the    |              |              |                |
|    | leopards?      |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 13 | My school's    | []           |           13 | 3003           |
|    | math club      |              |              |                |
|    | has 6 boys     |              |              |                |
|    | and 8 girls.   |              |              |                |
|    | I need to      |              |              |                |
|    | select a       |              |              |                |
|    | team to send   |              |              |                |
|    | to the state   |              |              |                |
|    | math           |              |              |                |
|    | competition.   |              |              |                |
|    | We want 6      |              |              |                |
|    | people on      |              |              |                |
|    | the team.      |              |              |                |
|    | In how many    |              |              |                |
|    | ways can I     |              |              |                |
|    | select the     |              |              |                |
|    | team without   |              |              |                |
|    | restrictions   |              |              |                |
|    | ?              |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 14 | Nathan will    | []           |           14 | 1/12           |
|    | roll two       |              |              |                |
|    | six-sided      |              |              |                |
|    | dice. What     |              |              |                |
|    | is the         |              |              |                |
|    | probability    |              |              |                |
|    | that he will   |              |              |                |
|    | roll a         |              |              |                |
|    | number less    |              |              |                |
|    | than three     |              |              |                |
|    | on the first   |              |              |                |
|    | die and a      |              |              |                |
|    | number         |              |              |                |
|    | greater than   |              |              |                |
|    | three on the   |              |              |                |
|    | second die?    |              |              |                |
|    | Express your   |              |              |                |
|    | answer as a    |              |              |                |
|    | common         |              |              |                |
|    | fraction.      |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 15 | A Senate       | []           |           15 | There are 56   |
|    | committee      |              |              | ways to form   |
|    | has 8          |              |              | such a subco   |
|    | Republicans    |              |              | mmittee.       |
|    | and 6          |              |              |                |
|    | Democrats.     |              |              |                |
|    | In how many    |              |              |                |
|    | ways can we    |              |              |                |
|    | form a         |              |              |                |
|    | subcommittee   |              |              |                |
|    | with 3         |              |              |                |
|    | Republicans    |              |              |                |
|    | and 2          |              |              |                |
|    | Democrats?     |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 16 | How many       | []           |           16 | 6              |
|    | different      |              |              |                |
|    | positive,      |              |              |                |
|    | four-digit     |              |              |                |
|    | integers can   |              |              |                |
|    | be formed      |              |              |                |
|    | using the      |              |              |                |
|    | digits 2, 2,   |              |              |                |
|    | 9 and 9?       |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 17 | I won a trip   | []           |           17 | 70 ways.       |
|    | for four to    |              |              |                |
|    | the Super      |              |              |                |
|    | Bowl.  I can   |              |              |                |
|    | bring three    |              |              |                |
|    | of my          |              |              |                |
|    | friends.  I    |              |              |                |
|    | have 8         |              |              |                |
|    | friends.  In   |              |              |                |
|    | how many       |              |              |                |
|    | ways can I     |              |              |                |
|    | form my        |              |              |                |
|    | Super Bowl     |              |              |                |
|    | party?         |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 18 | Determine      | []           |           18 | 60             |
|    | the number     |              |              |                |
|    | of ways to     |              |              |                |
|    | arrange the    |              |              |                |
|    | letters of     |              |              |                |
|    | the word       |              |              |                |
|    | MADAM.         |              |              |                |
+----+----------------+--------------+--------------+----------------+
| 19 | A palindrome   | []           |           19 | 900            |
|    | is a number    |              |              |                |
|    | that reads     |              |              |                |
|    | the same       |              |              |                |
|    | forwards and   |              |              |                |
|    | backwards,     |              |              |                |
|    | such as        |              |              |                |
|    | 3003. How      |              |              |                |
|    | many           |              |              |                |
|    | positive       |              |              |                |
|    | four-digit     |              |              |                |
|    | integers are   |              |              |                |
|    | palindromes?   |              |              |                |
+----+----------------+--------------+--------------+----------------+
```

:::

## フロー実行結果の評価

続いて、**評価フロー**を実行し、先ほどのチャットフローの実行結果を評価します。

評価フローは`promptflow/examples/flows/evaluation/eval-chat-math`に存在します。
評価フローへの入力は、正解 (groundtruth)として先ほどの`data.jsonl`の`answer`列が、予測 (prediction)として、先ほどのフロー (`$base_run_name`として定義したもの)実行結果の`outputs.answer`が与えられます。

```shell
cd ../evaluation
eval_run_name="eval_run"
pf run create --flow ./eval-chat-math --data ../chat/chat-math-variant/data.jsonl --column-mapping groundtruth='${data.answer}' prediction='${run.outputs.answer}' --stream --run $base_run_name --name $eval_run_name
```

実行結果：

```shell
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
    "run": "base_run"
}
```

`pf run show-metrics`コマンドを利用し評価フローのメトリックを取得します。

```shell
pf run show-metrics --name $eval_run_name
{
    "accuracy": 0.4,
    "error_rate": 0.6
}
```

`pf run visualize`を利用すると、ブラウザで`base_run`と`eval_run`の実行結果を一対一で可視化して比較できます。

```shell
pf run visualize --name "$base_run_name,$eval_run_name"
The HTML file is generated at '/tmp/pf-visualize-detail-dk5ldej6.html'.
Trying to view the result in a web browser...
Failed to visualize from the web browser, the HTML file locates at '/tmp/pf-visualize-detail-dk5ldej6.html'.
You can manually open it with your web browser, or try SDK to visualize it.
```

私が WSL 環境で実行しているためかブラウザの起動自体には失敗していますが、HTML ファイル (`/tmp/pf-visualize-detail-dk5ldej6.html`)は無事に生成されています。こちらをブラウザで表示します (私の場合は一度 WIndows 環境へコピーしてブラウザ実行しましたが、もっとうまいやり方はあるはずです)。

実行結果の可視化がされています。今回はフローへの 20 個の入力/出力があった中で、正解 (ground truth)と予測値 (チャットフロー実行結果)がそれぞれ一致していたかどうかの結果がテーブルで表示されています。また、右上ではグラフビューでフローを構成するノードが確認できます。

![](/images/promptflow-cli/flow-viz-1.png)

## プロンプトの改善とその評価

プロンプトの改善をしていきます。Chain-of-Thought (CoT)のアプローチを取り入れたプロンプトのフローを実行し、その回答精度を評価します。
`/chat/chat-math-variant`フォルダに 2 つの新たなプロンプトが用意されています。

```jinja2:chat_variant_1.jinja2
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

```jinja2:chat_variant_2.jinja2
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

```shell
cd ..
base_run_name="base_run_variant_"
eval_run_name="eval_run_variant_"
```

フローを実行します。長くて複雑なコマンドに見えますが、やっていることはこれまで行ったことの集合体です。

:::message
もし Azure OpenAI とのコネクション名をデフォルトから変更している場合は、それに合わせて`flow.dag.yaml`の`connection`パラメーターを変更してください。
:::

```shell
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

```shell
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

```shell
pf run visualize --name "${base_run_name}0,${eval_run_name}0,${base_run_name}1,${eval_run_name}1,${base_run_name}2,${eval_run_name}2"
```

出力された HTML ファイルをブラウザで開きます。

![](/images/promptflow-cli/flow-viz-2.png)

左上の Runs & Metrics パネル内の`evel_run_variant_2`が一番 accuracy と error rate の値が良いのが分かります。つまり`chat_variant_2.jinja2`が一番よい回答を生成したということです。

ただし、variant_2 の accuracy が 0.95 なのに対して、variant_1 の accuracy は 0.8 と比較的高い値になっています。variant_2 は variant_1 よりもかなり入力トークン数が多く、実行コストがかかるプロンプトです。実際のビジネス上問題がなければコストパフォーマンスのよい variant_1 のプロンプト利用も選択肢に入るでしょう。

なお、20 個あるテストデータの各データに対しての実行結果は、Outputs パネル内で確認できます。

# おわりに

Azure Machine Learning を使わずに、Prompt flow の CLI を利用してフローの作成から評価までを実行できました。
もちろん Azure Machine Learning を使うと今回行ったような Azure OpenAI などの外部サービスへの接続 (connection 設定)やフロー自体の編集も行いやすく便利です。ただフローデプロイの自動化を始め LLMOps を実現する上で YAML ベースのリソース定義や CLI の利用は不可欠だと思うので、ぜひ CLI も活用してください。

ちなみに [Prompt flow の VS Code 拡張機能](https://marketplace.visualstudio.com/items?itemName=prompt-flow.prompt-flow)を使うと今回みたいに HTML ファイルを生成してブラウザで表示、といった流れを行わずにフロー実行結果の可視化表示ができます。connection 設定やフローの編集など、今回 CLI を使って操作した内容が VS Code 上で簡単に利用できるので、Prompt flow を利用する際はぜひ利用してみてください。

![](/images/promptflow-cli/flow-vscode.png)
_VS Code 上で今回のフローを可視化した様子_
