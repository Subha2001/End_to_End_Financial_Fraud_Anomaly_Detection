pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }
    
    stages {
        stage('Cloning from Github Repo') {
            steps {
                script {
                    // Cloning Github repo
                    echo 'Cloning from Github Repo.....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'Subha_GitHub_Token', url: 'https://github.com/Subha2001/End_to_End_Financial_Fraud_Anomaly_Detection.git']])
                }
            }
        }

        stage('Setup Virtual Environment') {
            steps {
                script {
                    // Setup Virtual Environment
                    echo 'Setup Virtual Environment.....'
                    sh '''
                        python -m venv ${VENV_DIR}
                        . ${VENV_DIR}/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    '''
                }
            }
        }

        stage('Linting Code') {
            steps {
                script {
                    // Linting Code
                    echo 'Linting Code.....'
                    sh '''
                        set -e
                        . ${VENV_DIR}/bin/activate
                        pylint app.py --output=pylint-report.txt --exit-zero || echo "Pylint stage completed"
                    '''
                }
            }
        }

        stage('Trivy Scanning') {
            steps {
                script {
                    // Trivy Scanning
                    echo 'Trivy Scanning.....'
                    sh "trivy fs ./ --format table -o trivy-fs-report.html"
                }
            }
        }
    }
}