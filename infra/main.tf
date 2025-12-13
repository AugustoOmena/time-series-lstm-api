terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.5"
}

provider "aws" {
  region = var.region
}

# Obtém dados da conta atual
data "aws_caller_identity" "current" {}

# ---------------------------
# DATA SOURCES (Segredos)
# ---------------------------

data "aws_secretsmanager_secret" "dd_api_key" {
  name = "datadog/api_key"
}

# data "aws_secretsmanager_secret" "dd_app_key" {
#   name = "datadog/app_key"
# }

# ---------------------------
# ECR
# ---------------------------
resource "aws_ecr_repository" "app" {
  name = "fastapi-example"
  force_delete = true
}

# ---------------------------
# VPC + Subnets + Security Group
# ---------------------------
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_support   = true 
  enable_dns_hostnames = true
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

resource "aws_route_table_association" "subnet1_assoc" {
  subnet_id      = aws_subnet.subnet1.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "subnet2_assoc" {
  subnet_id      = aws_subnet.subnet2.id
  route_table_id = aws_route_table.public.id
}

resource "aws_subnet" "subnet1" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.region}a"
  map_public_ip_on_launch = true
}

resource "aws_subnet" "subnet2" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.region}b"
  map_public_ip_on_launch = true
}

resource "aws_security_group" "app_sg" {
  name        = "app-sg"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------
# ECS Cluster
# ---------------------------
resource "aws_ecs_cluster" "main" {
  name = "fastapi-cluster"
}

# ---------------------------
# ECS Task Definition
# ---------------------------
resource "aws_ecs_task_definition" "app" {
  family                   = "fastapi-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  execution_role_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/LabRole"
  task_role_arn      = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/LabRole"

  container_definitions = jsonencode([
    # --- Container 1: Sua Aplicação (FastAPI) ---
    {
      name      = "app"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com/fastapi-example:latest"
      essential = true
      
      # Força o uso do ddtrace-run caso o Dockerfile não tenha.
      command   = ["ddtrace-run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
      
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
      environment = [
        # Configurações para enviar traces para o agente local
        { name = "DD_AGENT_HOST", value = "localhost" },
        { name = "DD_TRACE_AGENT_PORT", value = "8126" },
        { name = "DD_SERVICE", value = "fastapi-app-lab" },
        { name = "DD_ENV", value = "lab-fiap" },
        { name = "DD_LOGS_INJECTION", value = "true" } # Relaciona Logs com Traces
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/example-task"
          awslogs-region        = "us-east-1"
          awslogs-stream-prefix = "app"
        }
      }
    },
    
    # --- Container 2: Datadog Agent (Sidecar) ---
    {
      name      = "datadog-agent"
      image     = "public.ecr.aws/datadog/agent:latest"
      essential = true
      cpu       = 256 # Reserva metade da CPU para o agente (máximo)
      memory    = 512 # Reserva metade da Memória
      
      environment = [
        { name = "ECS_FARGATE", value = "true" },
        { name = "DD_SITE", value = "datadoghq.com" },
        
        # Habilita coleta de processos e métricas do container
        { name = "DD_PROCESS_AGENT_ENABLED", value = "true" } 
      ]
      
      secrets = [
        {
          name      = "DD_API_KEY"
          valueFrom = data.aws_secretsmanager_secret.dd_api_key.arn
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/example-task"
          awslogs-region        = "us-east-1"
          awslogs-stream-prefix = "datadog"
        }
      }
    }
  ])
}

# ---------------------------
# CloudWatch
# ---------------------------
resource "aws_cloudwatch_log_group" "example" {
  name              = "/ecs/example-task"
  retention_in_days = 1
}

# ---------------------------
# ECS Service
# ---------------------------
resource "aws_ecs_service" "app" {
  name            = "fastapi-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = [aws_subnet.subnet1.id, aws_subnet.subnet2.id]
    security_groups = [aws_security_group.app_sg.id]
    assign_public_ip = true
  }
}
