from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(dag_id="real_estate_training_pipeline",
         description="Generate data, train/evaluate model, run tests",
         start_date=datetime(2024, 1, 1), schedule=None, catchup=False,
         tags=["mlops", "housing"]) as dag:
    generate_data = BashOperator(task_id="generate_data",
        bash_command="cd /opt/project && python -m src.generate_data")
    train_evaluate = BashOperator(task_id="train_evaluate",
        bash_command="cd /opt/project && python -m src.train")
    run_tests = BashOperator(task_id="run_tests",
        bash_command="cd /opt/project && python -m pytest -q tests")
    generate_data >> train_evaluate >> run_tests
