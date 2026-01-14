# Tech Challenge 4

### API de inferência para predição de ativos (ITUB4), implementada com FastAPI e implantada via Terraform em infraestrutura escalável AWS Fargate com monitoramento Datadog.

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

## 🧪 Testes Automatizados

Este projeto adota uma estratégia robusta de testes utilizando **Pytest**, focando na garantia da lógica de domínio e na integridade dos fluxos de dados sem depender de serviços externos instáveis.

### 🏗️ Arquitetura de Testes

A suíte de testes foi desenhada seguindo princípios de **Clean Architecture** e **S.O.L.I.D**, garantindo que a lógica de negócio (Domain Services) seja testada isoladamente da infraestrutura (APIs Externas, Banco de Dados).

- **Framework:** `pytest` + `pytest-mock`
- **Isolamento:** Uso extensivo de `unittest.mock` e `@patch` para simular chamadas ao Yahoo Finance e APIs de terceiros. Isso garante testes determinísticos, rápidos e que funcionam offline.
- **Fixtures:** Utilização de `conftest.py` para geração de massa de dados e DataFrames complexos, mantendo os arquivos de teste limpos.

### ⚙️ Configuração do Ambiente de Testes

Para manter a imagem Docker de produção leve (`slim`), as ferramentas de teste não são instaladas no container principal. Elas devem ser executadas em ambiente local ou em estágio de CI/CD.

1.  **Crie e ative seu ambiente virtual:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # ou
    .\venv\Scripts\activate   # Windows
    ```
2.  Instale as dependências de desenvolvimento: Nota: O arquivo requirements-dev.txt instala as libs do projeto + ferramentas de teste.

         pip install -r requirements-dev.txt

🚀 Executando os Testes
Para rodar a suíte completa com output detalhado:

         pytest -v

Para rodar apenas os testes unitários de domínio:

      pytest tests/unit -v

📂 Estrutura de Testes
Plaintext

tests/
├── conftest.py # Fixtures compartilhadas
(Dados Mockados)
├── unit/ # Testes de Unidade (Lógica de Negócio)
│ ├── test_avaluation_service.py
│ └── ...
└── integration/ # Testes de Integração (Endpoints FastAPI)
└── test_api_endpoints.py

---

### O que fazer agora (Dica Rápida):

Como mencionamos o arquivo `requirements-dev.txt` no README, lembre-se de criá-lo na raiz do projeto (caso não tenha feito ainda) com este conteúdo para que o comando funcione:

**Arquivo: `requirements-dev.txt`**

```text
-r requirements.txt
pytest==8.0.0
pytest-mock==3.12.0
httpx==0.26.0
```
