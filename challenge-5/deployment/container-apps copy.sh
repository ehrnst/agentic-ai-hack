#!/bin/bash
# Script to create Azure Container Registry, build/push image, and create Container App
set -e

RESOURCE_GROUP="" #FILL
SUBSCRIPTION_ID="" #FILL
LOCATION="swedencentral" 
ACR_NAME="" #FILL
IMAGE_NAME="insurance-orchestrator:latest"
CONTAINER_APP_NAME="" #FILL
DOCKERFILE_PATH="."

echo "🚀 Starting deployment to Azure Container Apps..."

# Create resource group (if it doesn't exist)
echo "📦 Ensuring resource group exists..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Build Docker image and push to ACR
echo "🔨 Building and pushing Docker image to ACR..."
az acr build --registry $ACR_NAME --image $IMAGE_NAME $DOCKERFILE_PATH

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)

# Create Container App Environment
CONTAINERAPPS_ENV="" #FILL
echo "🌍 Creating Container App Environment..."
az containerapp env create \
  --name $CONTAINERAPPS_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Create Container App with environment variables
echo "🚀 Creating Container App..."

az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINERAPPS_ENV \
  --image $ACR_LOGIN_SERVER/$IMAGE_NAME \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_NAME \
  --registry-password $(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv) \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 1 \
  --env-vars \
    AI_FOUNDRY_PROJECT_ENDPOINT="YOUR_AI_FOUNDRY_PROJECT_ENDPOINT" \
    MODEL_DEPLOYMENT_NAME="YOUR_MODEL_DEPLOYMENT_NAME" \
    COSMOS_ENDPOINT="YOUR_COSMOS_ENDPOINT" \
    COSMOS_KEY="YOUR_COSMOS_KEY" \
    AZURE_AI_CONNECTION_ID="YOUR_AZURE_AI_CONNECTION_ID" \
    AZURE_AI_SEARCH_INDEX_NAME="YOUR_SEARCH_INDEX_NAME" \
    SEARCH_SERVICE_NAME="YOUR_SEARCH_SERVICE_NAME" \
    SEARCH_SERVICE_ENDPOINT="YOUR_SEARCH_SERVICE_ENDPOINT" \
    SEARCH_ADMIN_KEY="YOUR_SEARCH_ADMIN_KEY"

# Enable managed identity
echo "🔐 Enabling managed identity..."
az containerapp identity assign \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --system-assigned

# Get the managed identity principal ID
PRINCIPAL_ID=$(az containerapp identity show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query principalId --output tsv)

echo "🔑 Managed Identity Principal ID: $PRINCIPAL_ID"

# Assign permissions to the managed identity
echo "🔐 Assigning permissions to managed identity..."
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services User" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP"

az containerapp ingress enable -n $CONTAINER_APP_NAME -g $RESOURCE_GROUP \
  --type external \
  --allow-insecure \
  --target-port 8000 \
  --transport auto

# Output info
echo "✅ Container App '$CONTAINER_APP_NAME' created successfully!"
echo "🔗 Image: $ACR_LOGIN_SERVER/$IMAGE_NAME"
echo "🔑 Managed Identity: $PRINCIPAL_ID"
echo "🌐 Access your Container App at: https://$CONTAINER_APP_NAME.azurecontainerapps.io/docs"
echo ""
echo "⚠️  SECURITY WARNING: Your app is now PUBLICLY ACCESSIBLE on the internet!"
echo "🔒  After testing"
echo "   • Disabling external ingress: az containerapp ingress disable -n $CONTAINER_APP_NAME -g $RESOURCE_GROUP"