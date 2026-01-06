# Simple Metaculus forecasting bot
This repository contains a simple bot meant to get you started with creating your own bot for the [AI Forecasting Tournament](https://www.metaculus.com/aib/). If you are looking for a more robust (but more complex) bot template/framework see [forecasting-tools](https://github.com/Metaculus/forecasting-tools)


## Quick start -> Fork and use Github Actions
The easiest way to use this repo is to fork it, enable github workflow/actions, and then set repository secrets. Then your bot will run every 30min, pick up new questions, and forecast on them. Automation is handled in the `.github/workflows/` folder. The `daily_run_simple_bot.yaml` file runs the simple bot every 30 min and will skip questions it has already forecasted on.

1) **Fork the repository**: Go to the [repository](https://github.com/Metaculus/metac-bot-template) and click 'fork'.
2) **Set secrets**: Go to `Settings -> Secrets and variables -> Actions -> New respository secret` and set API keys/Tokens as secrets. You will want to set your METACULUS_TOKEN. This will be used to post questions to Metaculus, and so you can use our OpenAI proxy (reach out to support@metaculus.com with your bot description to apply for credits. We are giving credits fairly generously to encourage participation).
3) **Enable Actions**: Go to 'Actions' then click 'Enable'. Then go to the 'Regularly forecast new questions' workflow, and click 'Enable'. To test if the workflow is working, click 'Run workflow', choose the main branch, then click the green 'Run workflow' button. This will check for new questions and forecast only on ones it has not yet successfully forecast on.

The bot should just work as is at this point. You can disable the workflow by clicking `Actions > Regularly forecast new questions > Triple dots > disable workflow`

As a note `GET_NEWS` is disabled by default, and you will need to edit this in `main.py` to enable searching the web. You will need a Asknews, Perplexity.ai, or Exa.ai API Key to enable searching. AskNews usage is free in Q1.

## Search Provider API Keys

### Getting Perplexity Set Up
Perplexity works as an internet powered LLM, and costs half a cent per search plus token costs. It is less customizable but generally cheaper.
1. Create an account on the free tier at www.perplexity.ai
2. Go to https://www.perplexity.ai/settings/account
3. Click "API" in the top bar
4. Click "Generate" in the "API Keys" section
5. Add funds to your account with the 'Buy Credits' button
6. Add it to the .env as `PERPLEXITY_API_KEY=your-key-here`

### Getting Exa Set Up
Exa is closer to a more traditional search provider. Exa takes in a search query and a list of filters and returns a list of websites. Each site returned can have scraped text, semantic higlights, AI summary, and more. By putting GPT on top of Exa, you can recreate Perplexity with more control. An implementation of this is available in the SmartSearcher of the forecasting-tools python package (though you will also need an OpenAI API key for this to work). Each Exa search costs half a cent per search plus a tenth of a cent per 'text-content' requested per site requested. Content items include: highlights from a source, summary of a source, or full text.
1. Make an account with Exa at Exa.ai
2. Go to https://dashboard.exa.ai/playground
3. Click on "API Keys" in the left sidebar
4. Create a new key
5. Go to 'Billing' in the left sidebar and add funds to your acount with the 'Top Up Balance'
6. Add it to the .env as `EXA_API_Key=your-key-here`

### Getting AskNews Setup
Metaculus is collaborating with AskNews to give free access to internet searches. Each registered bot builder gets 3k calls per month, 9k calls total for the entire tournament (please note that latest news requests (48 hours back) are 1 call and archive news requests are 5 calls). Bots have access to the /news endpoint only. To sign up:
1. make an account on AskNews (if you have not yet, https://my.asknews.app)
2. send the email address associated with your AskNews account to the email `rob [at] asknews.app`
3. in that email also send the name of your Metaculus Q1 bot
4. AskNews will make sure you have free calls and your account is ready to go for you to make API keys and get going
5. Generate your `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET` and add that to the .env

Your account will be active for the duration of the Q1 tournament. There is only one account allowed per participant.



## Run the bot locally
Clone the repository. Find your terminal and run the following commands:
```bash
git clone https://github.com/Metaculus/metac-bot-template.git
```

If you forked the repository first, you have to replace the url in the `git clone` command with the url to your fork. Just go to your forked repository and copy the url from the address bar in the browser.

### Installing dependencies
Make sure you have python and [poetry](https://python-poetry.org/docs/#installing-with-pipx) installed (poetry is a python package manager).

Inside the terminal, go to the directory you cloned the repository into and run the following command:
```bash
poetry install
```
to install all required dependencies.

### Setting environment variables

Running the bot requires various environment variables. If you run the bot locally, the easiest way to set them is to create a file called `.env` in the root directory of the repository (copy the `.env.template`).

### Running the bot

To run the simple bot, execute the following command in your terminal:
```bash
poetry run python main.py
```
Make sure to set the environment variables as described above and to set the parameters in the code to your liking. In particular, to submit predictions, make sure that `submit_predictions` is set to `True`.

## Fetching Community Forecasts from Metaculus API

When backtesting, you may want to compare your forecasts against the Metaculus community prediction. The API structure is non-trivial, so here's what we learned:

### Question Types & API Endpoints

**Standalone Questions** (binary, numeric, multiple choice)
- Endpoint: `GET /api/posts/{post_id}/`
- Community forecast location: `response.question.aggregations.unweighted.latest`
- For binary: `latest.centers[0]` = probability of "Yes"
- For numeric: `latest.forecast_values` = 201-point CDF
- For MC: `latest.centers` = list of probabilities per option

**Group Questions** (questions with sub-questions)
- The parent post has `response.group_of_questions` instead of `response.question`
- The parent post does NOT contain aggregations for sub-questions
- Sub-questions are identified by their `question_id` (different from `post_id`)

### The Solution for Group Questions

Use the download-data endpoint with the `sub_question` parameter:

```
GET /api/posts/{post_id}/download-data/?sub_question={question_id}
```

This returns a ZIP file containing CSVs. Parse `forecast_data.csv`:
- Look for rows where `Forecaster Username` is `"unweighted"` or `"recency_weighted"`
- For MC questions: `Probability Yes Per Category` column (parse as Python list)
- For numeric: `Continuous CDF` column (parse as 201-element list)

### Key Gotchas

1. **`recency_weighted.latest` is often empty** for resolved questions. Always fallback to `unweighted.latest`
2. **`/api/questions/{id}/` exists** but often returns `aggregations.*.latest: null` - use download-data instead
3. **Rate limits**: The API throttles at ~1000 requests/hour. Space your requests accordingly
4. **Group vs standalone**: Check for `response.question` vs `response.group_of_questions` to determine question type
5. **Question ID vs Post ID**: Group sub-questions have their own `question_id` which is different from the parent `post_id`

### Example Code

```python
import requests, zipfile, io, csv, ast

def get_community_forecast(post_id, question_id, question_type):
    """Fetch community forecast for any question, including group sub-questions."""
    headers = {"Authorization": f"Token {METACULUS_TOKEN}"}
    
    # Use download-data endpoint - works for all question types
    url = f"https://www.metaculus.com/api/posts/{post_id}/download-data/"
    if question_id != post_id:  # It's a sub-question
        url += f"?sub_question={question_id}"
    
    resp = requests.get(url, headers=headers, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    
    for name in z.namelist():
        if 'forecast_data' not in name:
            continue
        content = z.read(name).decode('utf-8')
        for row in csv.DictReader(io.StringIO(content)):
            if row.get('Forecaster Username') not in ['unweighted', 'recency_weighted']:
                continue
            
            if question_type == 'multiple_choice':
                return ast.literal_eval(row.get('Probability Yes Per Category', '[]'))
            elif question_type == 'numeric':
                return ast.literal_eval(row.get('Continuous CDF', '[]'))
            elif question_type == 'binary':
                return float(row.get('Probability Yes', 0))
    return None
```

