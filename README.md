# Tech Challenge 4

### API para prever preços da bolsa de valores, treinado sobre o dataset do ITAÚ.

# Como Executar

### Dockerfile (local)

Construir e rodar localmente:

            docker build -t fastapi-example .
            docker run -p 8000:8000 fastapi-example

Acesse:
http://localhost:8000

## Para fazer o deploy e usar na nuvem:

### Pré requisitos:

- ¹ Realize o Login na AWS (Explicado abaixo)
- Tenha instalado o Terraform
- Tenha instalado o AWS Cli
- Tenha instalado o Docker e ativo.

¹ Login na AWS (execute):

            nano ~/.aws/credentials

Substitua por suas credenciais atualizadas, feito.

### Passo 1: Pegue o Login do ECR

Use esse comando substituindo <AWS_ID> pelo seu ID da AWS:

            aws ecr get-login-password --region us-east-1 | \
            docker login --username AWS --password-stdin <AWS_ID>.dkr.ecr.us-east-1.amazonaws.com

### Passo 2: Criar o repositório ECR via Terraform

#### Importante: Estou considerando que você tem a AWS role "LabRole" fornecida com as permissões necessárias.

Se não tiver, use uma role com, no mínimo:

- execution role: anexar `AmazonECSTaskExecutionRolePolicy` (ECR pull + CloudWatch Logs)
- quem executa o Terraform precisa da permissão `iam:PassRole` para a role usada pela task

Tendo as permissões, continue:

            cd infra
            terraform init
            terraform apply -auto-approve

Ao executar com sucesso, vai retornar uma saída assim:
ecr_url = "<AWS_ID>.dkr.ecr.us-east-1.amazonaws.com/fastapi-example"

Copie o conteúdo entre aspas e use no passo 3.

### Nota sobre Datadog / secrets (simples)

Antes de executar `terraform apply`, crie no AWS Secrets Manager (ou SSM) os segredos que sua aplicação/Datadog vai usar (com os nomes: `datadog/api_key` e `datadog/app_key`).

Este repositório assume que os secrets já existem; crie-os via Console, CLI ou CI e só então rode o Terraform.

### Passo 3: Build + Push da imagem

Dicas:

1. Pegue a saída que você copiou no passo 2 e cole em <ECR_URL>;
2. Para executar, volte a raiz do projeto, onde se encontra o docker;
3. Se estiver executando no Macbook Apple Silicon usar buildx.

**Aviso: O Push pode demorar mais de 1h.**

**Antes do push:** Confirme que o login no ECR funcionou
(substitua <ECR_URL> pelo seu registry, ex: <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com — ou use a saída do Terraform (`ecr_url`) gerada no passo 2)

            aws ecr get-login-password --region us-east-1 \
            | docker login --username AWS --password-stdin <ECR_URL>

Se você estiver no **Windows** ou Linux (x86_64), execute:

            docker build -t fastapi-example .
            docker tag fastapi-example:latest <ECR_URL>:latest
            docker push <ECR_URL>:latest

Se você estiver no **Mac Apple Silicon** (M1/M2/M3), execute:

            docker buildx build \
            --platform=linux/amd64 \
            -t <ECR_URL>:latest \
            --push \
            .

Necessário para evitar o erro
exec format error
(porque o Fargate só roda imagens AMD64)

### Passo 4 (Último passo): Acessar

No Console AWS: Console AWS > ECS > Clusters > fastapi-cluster > Services > fastapi-service > Tasks > (selecionar task) > Network Interfaces → Public IP

http:<Public_IP>:8000/docs

Exemplo. 13.219.86.170:8000/docs
