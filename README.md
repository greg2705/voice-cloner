This document provides instructions on how to set up your development environment for working with Poetry and automated checks for code quality. 

This project should not be cloned but downloaded so that when you commit and push, you point your original project repository. Once you have downloaded this project, put your initial project in the src (or modify the project structure to fit your specific needs).

Then follow these steps to get started:

## Step 0: Install Poetry
Open your terminal or command prompt.
Run the following commands to install Poetry:
```python
pip install poetry
```
Create a new project using Poetry and set its name:

```python
poetry new my_project
```

Navigate into the newly created project directory:


```python
cd my_project
```

## Step 1: Configure HTTP Basic Authentication

Poetry is used to install packages instead of pip, then as you did for pip, you have to set it up to access the artifactory.
In the pyproject.toml this code sets the name and URL of your JFrog Artifactory repository:

```python
[[tool.poetry.source]]
name = "jfrog"
url="https://artifactory.cib.echonet/artifactory/api/pypi/rgmq-pypi/simple"
priority='primary'
```

Run the following command to configure HTTP basic authentication for JFrog Artifactory:

```python
poetry config http-basic.jfrog YOUR_UID YOUR_JFROG_TOKEN
```
Replace YOUR_UID and YOUR_JFROG_TOKEN with your actual UID and jfrog token.

## Step 2: Configure Certificates
Run the following command to configure certificates for JFrog Artifactory:

Download the certificate from `P:\DA_AllTeam\00 -Transfert TEMP\certificate`.

```python
poetry config certificates.jfrog.cert "C:\Users\a12345\OneDrive - BNP Paribas\Bureau\certificates\cacert-BNPP.pem"
```
Replace the path with the correct path to your certificate file.

## Step 3: Add Dependencies
Run the following command to add dependencies to your project:

```python
poetry add pandas
```

This will add Pandas as dependency. You can add the dependencies one by one or many at once by using:

```python
poetry add requirements.txt
```

This wil add the depencies to you project and make sure that there are no conflicts between the versions of the packages you want to use in your project, but the packages are not installed yet. To install it you have to run the command :

```python
poetry install 
```

Now the packages are installed and available for usage in an automatically generated virtual environment.

## Step 4: Activate Virtual Environment

Find the path of the virtual environment by running:

```python
poetry env info --path
```

Copy the path provided by the command. It should look like this : ```C:\Users\YOUR_UID\AppData\Local\pypoetry\Cache\virtualenvs.\POETRY_VIRTUAL_ENV_NAME```

Verify that the correct virtual environment is active by running:

```python
poetry env list
```

If the virtual envirnoment is not active, activate it manually by navigating to its directory and running the activation script:

```python
cd "C:\Users\YOUR_UID\AppData\Local\pypoetry\Cache\virtualenvs.\POETRY_VIRTUAL_ENV_NAME\Scripts\activate"
```

Then navigate back to your project directory:

```python
cd "YOUR_PROJECT_PATH"
```

In Visual Studio Code, open the settings file (settings.json) located in the .vscode folder.
Paste the python.exe (the one in the virtual env) path into the python.pythonPath setting:

```json
{
"python.pythonPath": "C:\\Users\\YOUR_UID\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\POETRY_VIRTUAL_ENV_NAME\\Scripts\\python.exe"
}
```
Save the changes.

If the virtual environment is not listed in the kernels, or if the .vscode folder doesn't apper in your project, set the python interpreter (python.exe) manually by digiting ```ctrl+shift+p```, then ```select interpreter``` and type the location of the python.exe of the virtual envitronment.

You should see in your terminal that the virtual env is activated. Now and you can use the packages that you have added as well as the ones which already were in the project (Ruff, Pytest, Commitizen, Poethepoet..)

## Step 5: Create your first tag
Create an initial tag for your project:
```python
git tag -a "0.0.1" -m "initial release"
```
This sets the first tag for future version increments.


## Step 6: Use the custom commands
Use the custom commands to take advantage of the automation:

The following command is used to run the unit tests and generate a coverage report.
```python
poe test 
```

This following will run quality code checks.
```python
poe check
```

The following will run code format checks.
```python
poe clean
```

The following command is equivalent to ```git status```
```python
poe status
```
The following command is equivalent to ```git add.```
```python
poe add
```
The following command must be used to commit and take advantage from the standard commit format. Run the command and follow the instructions to provide a commit description as clear as possible. 

Please note that each commit will trigger the pre-commit hooks that will check the quality of your code, you will not be allowed to commit up until all the errors are solved.
```python
poe commit
```
The following command will generate commit to create a bump, which is a new version of your projet, it will increment the project version in pyproject.toml, it will update the CHANGELOG.md whit your commit description and it will create a new git tag, so that you can get back to a previous tagged version of your code if needed.
```python
poe bump
```

The following command is equivalent to ```git push```

Please note that each push will trigger the pre-push hooks that will run the unit tests, you will not be allowed to push up until all the tests are successfull.
```python
poe push
```
The following command return all the tags that have created with ```poe bump```
```python
poe get_tags
```
The following command allows you to move to a specific version by adding the tag name you want to select. Exemple: ```poe change_version "1.0.8"```
```python
poe change_version
```
