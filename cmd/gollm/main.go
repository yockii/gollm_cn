package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	gollm "github.com/yockii/gollm_cn"
	"github.com/yockii/gollm_cn/optimizer"
	"github.com/yockii/gollm_cn/presets"
	"github.com/yockii/gollm_cn/utils"
)

func main() {
	// Existing flags
	promptType := flag.String("type", "raw", "提示类型 (raw, qa, cot, summarize, optimize)")
	verbose := flag.Bool("verbose", false, "显示详细输出，包括完整提示")
	provider := flag.String("provider", "", "LLM 提供者 (anthropic, openai, groq, mistral, ollama, cohere)")
	model := flag.String("model", "", "LLM 模型")
	temperature := flag.Float64("temperature", -1, "LLM 温度")
	maxTokens := flag.Int("max-tokens", 0, "LLM 最大 tokens")
	timeout := flag.Duration("timeout", 0, "LLM 超时")
	apiKey := flag.String("api-key", "", "指定提供者的 API 密钥")
	maxRetries := flag.Int("max-retries", 3, "API 调用最大重试次数")
	retryDelay := flag.Duration("retry-delay", time.Second*2, "重试之间的延迟")
	debugLevel := flag.String("debug-level", "warn", "调试级别 (debug, info, warn, error)")
	outputFormat := flag.String("output-format", "", "结构化响应的输出格式 (json)")

	// New flags for prompt optimization
	optimizeGoal := flag.String("optimize-goal", "提高提示的清晰度和有效性", "优化目标")
	optimizeIterations := flag.Int("optimize-iterations", 5, "优化迭代次数")
	optimizeMemory := flag.Int("optimize-memory", 2, "记住的先前迭代次数")

	flag.Parse()

	// Prepare configuration options
	configOpts := prepareConfigOptions(provider, model, temperature, maxTokens, timeout, apiKey, maxRetries, retryDelay, debugLevel)

	// Create LLM client with the specified options
	llmClient, err := gollm.NewLLM(configOpts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating LLM client: %v\n", err)
		os.Exit(1)
	}

	if len(flag.Args()) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <prompt>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	rawPrompt := strings.Join(flag.Args(), " ")
	ctx := context.Background()

	var response string
	var fullPrompt string

	switch *promptType {
	case "qa":
		response, err = presets.QuestionAnswer(ctx, llmClient, rawPrompt)
	case "cot":
		response, err = presets.ChainOfThought(ctx, llmClient, rawPrompt)
	case "summarize":
		response, err = presets.Summarize(ctx, llmClient, rawPrompt)
	case "optimize":
		optimizer := optimizer.NewPromptOptimizer(
			llmClient,
			utils.NewDebugManager(
				llmClient.GetLogger(),
				utils.DebugOptions{LogPrompts: true, LogResponses: true}),
			llmClient.NewPrompt(rawPrompt),
			*optimizeGoal,
			optimizer.WithIterations(*optimizeIterations),
			optimizer.WithMemorySize(*optimizeMemory),
		)
		optimizedPrompt, err := optimizer.OptimizePrompt(ctx)
		if err == nil {
			response = optimizedPrompt.Input
			fullPrompt = fmt.Sprintf("Initial Prompt: %s\nOptimization Goal: %s\nMemory Size: %d", rawPrompt, *optimizeGoal, *optimizeMemory)
		}
	default:
		prompt := gollm.NewPrompt(rawPrompt)
		if *outputFormat == "json" {
			prompt.Apply(gollm.WithOutput("Please provide your response in JSON format."))
		}
		response, err = llmClient.Generate(ctx, prompt, gollm.WithJSONSchemaValidation())
		fullPrompt = prompt.String()
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating response: %v\n", err)
		os.Exit(1)
	}

	printResponse(*verbose, *promptType, fullPrompt, rawPrompt, response, *outputFormat)
}

func prepareConfigOptions(provider, model *string, temperature *float64, maxTokens *int, timeout *time.Duration, apiKey *string, maxRetries *int, retryDelay *time.Duration, debugLevel *string) []gollm.ConfigOption {
	var configOpts []gollm.ConfigOption

	if *provider != "" {
		configOpts = append(configOpts, gollm.SetProvider(*provider))
	}
	if *model != "" {
		configOpts = append(configOpts, gollm.SetModel(*model))
	}
	if *temperature != -1 {
		configOpts = append(configOpts, gollm.SetTemperature(*temperature))
	}
	if *maxTokens != 0 {
		configOpts = append(configOpts, gollm.SetMaxTokens(*maxTokens))
	}
	if *timeout != 0 {
		configOpts = append(configOpts, gollm.SetTimeout(*timeout))
	}
	if *apiKey != "" {
		configOpts = append(configOpts, gollm.SetAPIKey(*apiKey))
	}
	configOpts = append(configOpts, gollm.SetMaxRetries(*maxRetries))
	configOpts = append(configOpts, gollm.SetRetryDelay(*retryDelay))
	configOpts = append(configOpts, gollm.SetLogLevel(gollm.LogLevel(getLogLevel(*debugLevel))))

	return configOpts
}

func printResponse(verbose bool, promptType, fullPrompt, rawPrompt, response, outputFormat string) {
	if verbose {
		if fullPrompt == "" {
			fullPrompt = rawPrompt // For qa, cot, and summarize, we don't have access to the full prompt
		}
		fmt.Printf("Prompt Type: %s\nFull Prompt:\n%s\n\nResponse:\n---------\n", promptType, fullPrompt)
	}

	if outputFormat == "json" {
		var jsonResponse interface{}
		err := json.Unmarshal([]byte(response), &jsonResponse)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing JSON response: %v\n", err)
			fmt.Println(response) // Print raw response if JSON parsing fails
		} else {
			jsonPretty, _ := json.MarshalIndent(jsonResponse, "", "  ")
			fmt.Println(string(jsonPretty))
		}
	} else {
		fmt.Println(response)
	}
}

func getLogLevel(level string) gollm.LogLevel {
	switch strings.ToLower(level) {
	case "debug":
		return gollm.LogLevelDebug
	case "info":
		return gollm.LogLevelInfo
	case "error":
		return gollm.LogLevelError
	default:
		return gollm.LogLevelWarn
	}
}
