pipeline {
    agent any
    
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
    }
}