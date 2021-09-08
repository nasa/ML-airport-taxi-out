
def shouldBumpVersion = true
def shouldBuild = true

pipeline {
   agent any
   environment {
       PATH = "$PATH:$PIPELINE_SCRIPTS"
   }
   parameters {
      choice(
         choices: ['patch', 'minor', 'major'],
         description: 'Release type',
         name: 'RELEASE_TYPE'
      )
   }
   stages {
        stage ('Analyze') {
            steps {             
                script {
                    result = sh (script: "git log -1 | grep '.*Bump version.*'", returnStatus: true)
                    
                    if (result == 0) {
                       echo ("Detecting previous build was release, skipping version bumping")
                       shouldBumpVersion = false
                       shouldBuild = false
                    }
                    
                    if (! shouldBumpVersion)
                    {                       
                       if (BUILD_TYPE == "none" || BUILD_TYPE == "release")
                       {
                          shouldBuild = true;
                       }
                    }
                    
                    echo("Should bump version: ${shouldBumpVersion}")
                    echo("Should build: ${shouldBuild}");
                }
            }
        }
       stage ('Bump Development Version') {
           when {
                expression {
                    shouldBumpVersion
                }
               environment name: 'BUILD_TYPE', value: 'dev'
           }
           steps {
               sh 'updateDevVersion.sh'
               sh 'pushChanges.sh'
           }
       }
       stage ('Release') {
           when {
               environment name: 'BUILD_TYPE', value: 'release'
           }
           steps {
               sh 'updateReleaseVersion.sh'
               sh 'pushChanges.sh'
           }
       }
       stage ('Build') {
           when {
             expression {
                 shouldBuild
             }
           }
           steps {
               sh 'buildPythonProject.sh'
           }
           post {
               success {
                   archiveArtifacts 'dist/*'
               }
           }
       }
       stage ('Generate Environment') {
            when {
               expression {
                 shouldBuild
               }
            }
            steps {
                sh 'packageEnvironment.sh'
                sh 'bundleArtifact.sh'
            }
            post {
                success {
                    archiveArtifacts '*.tar.gz'
                }
            }
       }
       stage ('Post Release') {
           when {
               environment name: 'BUILD_TYPE', value: 'release'
           }
           parallel{
              stage('Update Major Version') {
                when {
                    expression{ params.RELEASE_TYPE == 'major' }
                }
                steps {
                    sh 'updateMajorVersion.sh'
                    sh 'pushChanges.sh'
                    sh 'buildPythonProject.sh'
                }
              }
              stage('Update Minor Version') {
                when {
                    expression{ params.RELEASE_TYPE == 'minor' }
                }
                steps {
                    sh 'updateMinorVersion.sh'
                    sh 'pushChanges.sh'
                    sh 'buildPythonProject.sh'
                }
              }
              stage('Update Patch Version') {
                when {
                    expression{ params.RELEASE_TYPE == 'patch' }
                }
                steps {
                    sh 'updatePatchVersion.sh'
                    sh 'pushChanges.sh'
                    sh 'buildPythonProject.sh'
                }
              }
           }
           post {
               success {
                   archiveArtifacts 'dist/*'
               }
           }
       }
       stage ('Deploy') {
           when {
             expression {
                 shouldBuild
             }
           }
           steps {
               sh 'deployPythonProject.sh ${ARTIFACT_REPOSITORY}'
           }
       }
   }
}