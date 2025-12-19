# Tech Challenge 4

### API para prever preços da bolsa de valores, treinado sobre o dataset do ITAÚ.

# Como Executar

### Dockerfile (local)

Construir e rodar localmente:

            docker build -t fastapi-example .
            docker run -p 8000:8000 fastapi-example

Acesse:
http://localhost:8000

## Deploy na Nuvem (AWS)

### Pré requisitos:

- Contas: AWS (com permissões de Admin ou LabRole) e Datadog.

- Ferramentas: Terraform, AWS CLI, Docker.

- Login AWS: Configure suas credenciais em ~/.aws/credentials.

### Passo 1: Autenticação no ECR

Antes de tudo, o Docker precisa de permissão para falar com a AWS. Substitua <AWS_ID> pelo seu ID da conta:

            aws ecr get-login-password --region us-east-1 | \
            docker login --username AWS --password-stdin <AWS_ID>.dkr.ecr.us-east-1.amazonaws.com

### Passo 2: Configuração de Segredos (Datadog)

O Terraform espera que os segredos do Datadog já existam no AWS Secrets Manager. Execute os comandos abaixo para criá-los via CLI (mais rápido e seguro que o console):

            aws secretsmanager create-secret --name "datadog/api_key" \
            --description "Datadog API Key" --secret-string "COLE_SUA_API_KEY_AQUI"

            aws secretsmanager create-secret --name "datadog/app_key" \
            --description "Datadog APP Key" --secret-string "COLE_SUA_APP_KEY_AQUI"

### Passo 3: Provisionamento da Infraestrutura (Terraform)

Agora, vamos subir o ECR, Cluster ECS e Fargate.

#### Nota: Este projeto utiliza a role LabRole. Certifique-se de que ela possui as políticas AmazonECSTaskExecutionRolePolicy e iam:PassRole.

Se não tiver, use uma role com, no mínimo:

            cd infra
            terraform init
            terraform apply -auto-approve

**Importante**: Ao finalizar, copie a ecr_url exibida no terminal. Você a usará como <ECR_URL> no próximo passo.

### Nota sobre Datadog / secrets (simples)

Antes de executar `terraform apply`, crie no AWS Secrets Manager (ou SSM) os segredos que sua aplicação/Datadog vai usar (com os nomes: `datadog/api_key` e `datadog/app_key`).

Este repositório assume que os secrets já existem; crie-os via Console, CLI ou CI e só então rode o Terraform.

### Passo 4: Build + Push da imagem

Volte para a raiz do projeto. O push pode demorar dependendo da sua conexão.

Se você estiver no **Windows** ou Linux (x86_64), execute:

            docker build -t fastapi-example .
            docker tag fastapi-example:latest <ECR_URL>:latest
            docker push <ECR_URL>:latest

**Para Mac Apple Silicon (M1/M2/M3):** Essencial para evitar o erro exec format error no Fargate.

            docker buildx build --platform=linux/amd64 -t <ECR_URL>:latest --push .

### Passo 5: Acesso e Monitoramento

Aguarde alguns minutos para o serviço estabilizar no ECS.

Vá ao Console AWS > ECS > Clusters > fastapi-cluster.

Clique em Services > fastapi-service > aba Tasks.

Abra a Task ativa e copie o Public IP em Network Interfaces.

Acesse: http://<PUBLIC_IP>:8000/docs
