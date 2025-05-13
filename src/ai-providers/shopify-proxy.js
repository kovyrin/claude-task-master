/**
 * src/ai-providers/shopify-proxy.js
 *
 * Implementation for interacting with Shopify AI Proxy which provides access to
 * various AI models like OpenAI and Anthropic through a unified proxy interface.
 */
import { createOpenAI } from '@ai-sdk/openai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { generateText, streamText, generateObject } from 'ai';
import { log } from '../../scripts/modules/utils.js';

/**
 * Helper function to format messages appropriately for different models
 */
function formatMessages(messages, vendorType) {
  // If there are no messages, return an empty array
  if (!messages || !Array.isArray(messages)) return [];

  // Format messages based on vendor type
  return messages.map(msg => {
    const role = msg.role?.toLowerCase();

    // For anthropic, convert system messages to annotated user messages
    if (vendorType === 'anthropic' && role === 'system') {
      return { role: 'user', content: `<system>${msg.content}</system>` };
    }

    // For all models, normalize roles
    if (role === 'developer' || role === 'agent') {
      return { role: 'system', content: msg.content };
    }

    return msg;
  });
}

/**
 * Parse the model ID to extract vendor and model name
 * @param {string} modelId - Model ID in format "vendor:model_name"
 * @returns {object} Object with vendor and modelName properties
 */
function parseModelId(modelId) {
  if (!modelId) throw new Error('Model ID is required');

  const parts = modelId.split(':');
  if (parts.length === 2) {
    return { vendor: parts[0], modelName: parts[1] };
  }

  // Default to openai for unspecified formats
  return { vendor: 'openai', modelName: modelId };
}

/**
 * Get the appropriate AI client based on vendor type
 * @param {string} vendor - The vendor name ('openai' or 'anthropic')
 * @param {string} apiKey - The API key
 * @param {string} baseUrl - The base URL for the Shopify Proxy
 * @returns {object} The configured AI client
 */
function getClient(vendor, apiKey, baseUrl) {
  if (!apiKey) {
    throw new Error(`API key is required for ${vendor} client`);
  }

  const vendorBaseUrl = `${baseUrl}/vendors/${vendor}`;
  log('debug', `Using vendor base URL: ${vendorBaseUrl}`);

  if (vendor === 'anthropic') {
    return createAnthropic({
      apiKey,
      baseURL: `${vendorBaseUrl}/v1`,
      headers: {
        'anthropic-version': '2023-06-01',
        'anthropic-beta': 'output-128k-2025-02-19'
      }
    });
  } else {
    // Default to OpenAI
    return createOpenAI({
      apiKey,
      baseURL: `${vendorBaseUrl}/v1`
    });
  }
}

/**
 * Generates text using Shopify Proxy
 *
 * @param {object} params - Parameters for text generation
 * @param {string} params.apiKey - The API key
 * @param {string} params.modelId - The model ID in format "vendor:model_name"
 * @param {Array<object>} params.messages - The messages array
 * @param {number} [params.maxTokens] - Maximum tokens for the response
 * @param {number} [params.temperature] - Temperature for generation
 * @param {string} [params.baseUrl] - Override the default base URL
 * @returns {Promise<string>} The generated text content
 */
export async function generateShopifyProxyText({
  apiKey,
  modelId,
  messages,
  maxTokens = 1000,
  temperature = 0.7,
  baseUrl = 'https://proxy.shopify.ai'
}) {
  log('debug', `Generating Shopify Proxy text with model: ${modelId}`);

  if (!apiKey) throw new Error('API key is required');
  if (!modelId) throw new Error('Model ID is required');
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    throw new Error('Invalid or empty messages array');
  }

  try {
    const { vendor, modelName } = parseModelId(modelId);
    const client = getClient(vendor, apiKey, baseUrl);
    const formattedMessages = formatMessages(messages, vendor);

    log('debug', `Using vendor: ${vendor}, model: ${modelName}`);

    const result = await generateText({
      model: client(modelName),
      messages: formattedMessages,
      maxTokens,
      temperature
    });

    log(
      'debug',
      `Shopify Proxy generateText result received. Tokens: ${result.usage?.completionTokens || 'unknown'}/${result.usage?.promptTokens || 'unknown'}`
    );

    return result.text;
  } catch (error) {
    log('error', `Shopify Proxy generateText failed: ${error.message}`);
    throw error;
  }
}

/**
 * Streams text using Shopify Proxy
 *
 * @param {object} params - Parameters for text streaming
 * @param {string} params.apiKey - The API key
 * @param {string} params.modelId - The model ID in format "vendor:model_name"
 * @param {Array<object>} params.messages - The messages array
 * @param {number} [params.maxTokens] - Maximum tokens for the response
 * @param {number} [params.temperature] - Temperature for generation
 * @param {string} [params.baseUrl] - Override the default base URL
 * @returns {Promise<object>} The stream result object
 */
export async function streamShopifyProxyText({
  apiKey,
  modelId,
  messages,
  maxTokens = 1000,
  temperature = 0.7,
  baseUrl = 'https://proxy.shopify.ai'
}) {
  log('debug', `Streaming Shopify Proxy text with model: ${modelId}`);

  if (!apiKey) throw new Error('API key is required');
  if (!modelId) throw new Error('Model ID is required');
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    throw new Error('Invalid or empty messages array');
  }

  try {
    const { vendor, modelName } = parseModelId(modelId);
    const client = getClient(vendor, apiKey, baseUrl);
    const formattedMessages = formatMessages(messages, vendor);

    log('debug', `Using vendor: ${vendor}, model: ${modelName} for streaming`);

    const stream = await streamText({
      model: client(modelName),
      messages: formattedMessages,
      maxTokens,
      temperature
    });

    log('debug', `Shopify Proxy stream initiated successfully`);
    return stream;
  } catch (error) {
    log('error', `Shopify Proxy streamText failed: ${error.message}`, error.stack);
    throw error;
  }
}

/**
 * Generates a structured object using Shopify Proxy
 *
 * @param {object} params - Parameters for object generation
 * @param {string} params.apiKey - The API key
 * @param {string} params.modelId - The model ID in format "vendor:model_name"
 * @param {Array<object>} params.messages - The messages array
 * @param {import('zod').ZodSchema} params.schema - The Zod schema for the object
 * @param {string} params.objectName - A name for the object/tool
 * @param {number} [params.maxTokens] - Maximum tokens for the response
 * @param {number} [params.temperature] - Temperature for generation
 * @param {number} [params.maxRetries] - Max retries for validation/generation
 * @param {string} [params.baseUrl] - Override the default base URL
 * @returns {Promise<object>} The generated object matching the schema
 */
export async function generateShopifyProxyObject({
  apiKey,
  modelId,
  messages,
  schema,
  objectName = 'generated_object',
  maxTokens = 1000,
  temperature = 0.2,
  maxRetries = 3,
  baseUrl = 'https://proxy.shopify.ai'
}) {
  log('debug', `Generating Shopify Proxy object (${objectName}) with model: ${modelId}`);

  if (!apiKey) throw new Error('API key is required');
  if (!modelId) throw new Error('Model ID is required');
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    throw new Error('Invalid or empty messages array');
  }
  if (!schema) throw new Error('Schema is required');
  if (!objectName) throw new Error('Object name is required');

  try {
    const { vendor, modelName } = parseModelId(modelId);
    const client = getClient(vendor, apiKey, baseUrl);
    const formattedMessages = formatMessages(messages, vendor);

    log('debug', `Using vendor: ${vendor}, model: ${modelName} for object generation`);

    // Create base parameters for all models
    const generationParams = {
      model: client(modelName),
      messages: formattedMessages,
      schema,
      maxTokens,
      temperature,
      maxRetries
    };

    // Adjust parameters based on vendor
    if (vendor === 'anthropic') {
      // For Anthropic, we use the tool mode with the specific schema format
      generationParams.mode = 'tool';
      generationParams.tool = {
        name: objectName,
        description: `Generate a ${objectName} based on the prompt.`,
        input_schema: schema
      };
    } else {
      // For OpenAI and other vendors, we use json mode
      generationParams.mode = 'json';
    }

    const result = await generateObject(generationParams);

    log(
      'debug',
      `Shopify Proxy generateObject result received. Tokens: ${result.usage?.completionTokens || 'unknown'}/${result.usage?.promptTokens || 'unknown'}`
    );

    return result.object;
  } catch (error) {
    log('error', `Shopify Proxy generateObject (${objectName}) failed: ${error.message}`);
    throw error;
  }
}
