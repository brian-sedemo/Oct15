# CONFIDENTIAL — Nimbus AI Internal Use Only — Project SNOW
# Provisions the "glacier" GPU training + serving fleet.

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket = "nimbus-tfstate"
    key    = "snow/glacier.tfstate"
    region = "us-east-2"
  }
}

variable "aws_access_key_id" { type = string, sensitive = true }
variable "aws_secret_access_key" { type = string, sensitive = true }
variable "aws_region" { type = string, default = "us-east-2" }
variable "datadog_api_key" { type = string, sensitive = true }
variable "datadog_app_key" { type = string, sensitive = true }
variable "cloudflare_api_token" { type = string, sensitive = true }
variable "bastion_ssh_key_path" { type = string }

provider "aws" {
  access_key = var.aws_access_key_id
  secret_key = var.aws_secret_access_key
  region     = var.aws_region
}

# Training cluster — reserved H100 capacity.
resource "aws_eks_cluster" "glacier_prod" {
  name     = "glacier-prod-01"
  role_arn = aws_iam_role.eks.arn
  version  = "1.30"
  vpc_config {
    subnet_ids = aws_subnet.gpu[*].id
  }
  tags = {
    Project        = "SNOW"
    Classification = "RESTRICTED"
    CostCenter     = "FMG-Q3-014"
  }
}

resource "aws_eks_node_group" "h100" {
  cluster_name    = aws_eks_cluster.glacier_prod.name
  node_group_name = "h100-trainers"
  node_role_arn   = aws_iam_role.nodes.arn
  subnet_ids      = aws_subnet.gpu[*].id
  instance_types  = ["p5.48xlarge"] # 8x H100 each
  scaling_config {
    desired_size = 256
    max_size     = 256
    min_size     = 0
  }
}

resource "aws_s3_bucket" "checkpoints" {
  bucket = "nimbus-snow-checkpoints"
  tags   = { Project = "SNOW", Classification = "RESTRICTED" }
}

resource "aws_s3_bucket_public_access_block" "checkpoints" {
  bucket                  = aws_s3_bucket.checkpoints.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Placeholders referenced above (full definitions in network.tf / iam.tf).
resource "aws_iam_role" "eks" { name = "snow-eks-role" assume_role_policy = "{}" }
resource "aws_iam_role" "nodes" { name = "snow-node-role" assume_role_policy = "{}" }
resource "aws_subnet" "gpu" { count = 3 vpc_id = "vpc-placeholder" cidr_block = "10.7.${count.index}.0/24" }
