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
  region  = var.region
  profile = "pessoal"
}

# Obtém dados da conta atual
data "aws_caller_identity" "current" {}

# ---------------------------
# IAM ROLE (A CORREÇÃO DO PROBLEMA)
# ---------------------------
# Criamos uma Role que o ECS tem permissão para "assumir"
resource "aws_iam_role" "ecs_execution_role" {
  name = "ecs_task_execution_role_pessoal"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Damos permissão de ADMIN para essa Role (Para não ter erro de permissão no seu lab pessoal)
resource "aws_iam_role_policy_attachment" "ecs_admin_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

# ---------------------------
# DATA SOURCES (Segredos)
# ---------------------------
data "aws_secretsmanager_secret" "dd_api_key" {
  name = "datadog/api_key"
}

# ---------------------------
# ECR
# ---------------------------
resource "aws_ecr_repository" "app" {
  name         = "fastapi-example"
  force_delete = true
}

# ---------------------------
# VPC + Subnets
# ---------------------------
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
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

resource "aws_route_table_association" "subnet1_assoc" {
  subnet_id      = aws_subnet.subnet1.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "subnet2_assoc" {
  subnet_id      = aws_subnet.subnet2.id
  route_table_id = aws_route_table.public.id
}

# ---------------------------
# SECURITY GROUPS
# ---------------------------
resource "aws_security_group" "alb_sg" {
  name        = "alb-sg"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP da Internet"
    from_port   = 80
    to_port     = 80
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

resource "aws_security_group" "app_sg" {
  name        = "app-sg"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Trafego vindo do ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------
# LOAD BALANCER
# ---------------------------
resource "aws_lb" "main" {
  name               = "fastapi-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = [aws_subnet.subnet1.id, aws_subnet.subnet2.id]
}

resource "aws_lb_target_group" "app" {
  name        = "fastapi-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    # 1. Mudamos para /docs pois é garantido que o FastAPI responde 200 nela
    path                = "/docs" 
    
    # 2. Aceita códigos de sucesso (200 OK)
    matcher             = "200-299"
    
    # 3. PACIÊNCIA: Checa a cada 60s (menos estresse no container)
    interval            = 60
    
    # 4. Espera até 20s pela resposta antes de considerar falha
    timeout             = 20
    
    # 5. Precisa falhar 5 vezes seguidas para o ECS matar a tarefa (dá 5 min de chance)
    unhealthy_threshold = 5
    
    # 6. Basta 2 sucessos para liberar o tráfego
    healthy_threshold   = 2
  }
}

resource "aws_lb_listener" "front_end" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# ---------------------------
# ECS Cluster & Task
# ---------------------------
resource "aws_ecs_cluster" "main" {
  name = "fastapi-cluster"
}

resource "aws_ecs_task_definition" "app" {
  family                   = "fastapi-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  # AQUI ESTAVA O ERRO: Agora usamos a Role criada pelo Terraform
  execution_role_arn = aws_iam_role.ecs_execution_role.arn
  task_role_arn      = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "app"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com/fastapi-example:v2"
      essential = true
      cpu       = 10
      
      # Timeout para dar tempo da imagem chegar
      startTimeout = 120
      
      command   = ["ddtrace-run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
      
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
      environment = [
        { name = "DD_AGENT_HOST", value = "localhost" },
        { name = "DD_TRACE_AGENT_PORT", value = "8126" },
        { name = "DD_SERVICE", value = "fastapi-app-lab" },
        { name = "DD_ENV", value = "lab-fiap" },
        { name = "DD_LOGS_INJECTION", value = "true" }
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
    {
      name      = "datadog-agent"
      image     = "public.ecr.aws/datadog/agent:latest"
      essential = true

      environment = [
        { name = "ECS_FARGATE", value = "true" },
        { name = "DD_SITE", value = "datadoghq.com" },
        { name = "DD_PROCESS_AGENT_ENABLED", value = "false" }
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

  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 100
  }

  network_configuration {
    subnets          = [aws_subnet.subnet1.id, aws_subnet.subnet2.id]
    security_groups  = [aws_security_group.app_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.front_end]
}

# ---------------------------
# OUTPUTS
# ---------------------------
output "url_api" {
  description = "Link fixo da API"
  value       = "http://${aws_lb.main.dns_name}"
}