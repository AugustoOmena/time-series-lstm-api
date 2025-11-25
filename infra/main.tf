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
  cpu                      = "256"
  memory                   = "512"

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  execution_role_arn = "arn:aws:iam::610520926426:role/LabRole"
  task_role_arn      = "arn:aws:iam::610520926426:role/LabRole"

  container_definitions = jsonencode([
    {
      name      = "app"
      image = "610520926426.dkr.ecr.us-east-1.amazonaws.com/fastapi-example:latest"
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/example-task"
          awslogs-region        = "us-east-1" # Substitua pela sua região
          awslogs-stream-prefix = "ecs"
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
