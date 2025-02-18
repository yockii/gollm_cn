// Package presets provides utilities for enhancing Language Learning Model interactions
// with specific reasoning patterns and structured data extraction capabilities.
package presets

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	gollm "github.com/yockii/gollm_cn"
)

// ExtractStructuredData extracts structured data from unstructured text by mapping it
// to a strongly-typed Go struct. It uses JSON schema validation to ensure the extracted
// data matches the expected structure and constraints.
//
// The function is generic and can work with any struct type that can be represented
// as JSON and validated using the standard validation tags. It automatically generates
// a JSON schema from the provided type T and instructs the LLM to extract information
// that matches this schema.
//
// Type Parameters:
//   - T: The target struct type that defines the structure of the data to extract
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - l: LLM instance to use for extraction
//   - text: The unstructured text to extract information from
//   - opts: Optional prompt configuration options
//
// Returns:
//   - *T: Pointer to the extracted and validated data structure
//   - error: Any error encountered during extraction, parsing, or validation
//
// Example usage with a simple person struct:
//
//	type PersonInfo struct {
//	    Name       string   `json:"name" validate:"required"`
//	    Age        int      `json:"age" validate:"required,gte=0,lte=150"`
//	    Occupation string   `json:"occupation" validate:"required"`
//	    Hobbies    []string `json:"hobbies" validate:"required,min=1,max=5"`
//	}
//
//	text := `John Smith is a 32-year-old software engineer from Seattle.
//	         He enjoys hiking, photography, and playing guitar in his free time.`
//
//	person, err := ExtractStructuredData[PersonInfo](ctx, llm, text,
//	    gollm.WithMaxTokens(300),
//	    gollm.WithTemperature(0.2),
//	)
//
// Example usage with a complex nested struct:
//
//	type ComplexPerson struct {
//	    Name          string   `json:"name" validate:"required"`
//	    Age           int      `json:"age" validate:"required,gte=0,lte=150"`
//	    Occupation    string   `json:"occupation" validate:"required"`
//	    City          string   `json:"city" validate:"required"`
//	    Country       string   `json:"country" validate:"required"`
//	    FavoriteColor string   `json:"favoriteColor" validate:"required"`
//	    Hobbies       []string `json:"hobbies" validate:"required,min=1,max=5"`
//	    Education     string   `json:"education" validate:"required"`
//	    PetName       string   `json:"petName" validate:"required"`
//	    LuckyNumber   int      `json:"luckyNumber" validate:"required,gte=1,lte=100"`
//	}
//
//	text := `Sarah Johnson holds a Ph.D. in Physics and works as a Research Scientist
//	         at a national laboratory in Berkeley, USA. At 29, she balances her time
//	         between quantum experiments and taking care of her cat "Einstein". Her
//	         apartment is decorated in shades of purple, her favorite color. She loves
//	         rock climbing, quantum computing research, and playing the violin. Sarah
//	         considers 42 her lucky number.`
//
//	person, err := ExtractStructuredData[ComplexPerson](ctx, llm, text,
//	    gollm.WithMaxTokens(500),
//	    gollm.WithTemperature(0.1),
//	    gollm.WithDirectives(
//	        "Extract all available information accurately",
//	        "Ensure numeric values are within valid ranges",
//	        "Leave fields as null if information is not clearly stated",
//	    ),
//	)
//
// The function performs the following steps:
// 1. Generates a JSON schema from the target type T
// 2. Creates a prompt that includes the input text and schema
// 3. Instructs the LLM to extract information matching the schema
// 4. Parses and validates the LLM's response
// 5. Returns the validated structured data
//
// Common validation tags supported:
//   - required: Field must be present and non-empty
//   - min,max: Array length constraints
//   - gte,lte: Numeric range constraints
//   - oneof: Value must be one of the specified options
//   - email: Must be valid email format
//   - url: Must be valid URL format
//
// Error handling:
//   - Schema generation errors
//   - LLM response generation errors
//   - JSON parsing errors
//   - Validation constraint violations
func ExtractStructuredData[T any](ctx context.Context, l gollm.LLM, text string, opts ...gollm.PromptOption) (*T, error) {
	// Validate input
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}
	if l == nil {
		return nil, fmt.Errorf("LLM instance cannot be nil")
	}
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	var zero T
	schema, err := gollm.GenerateJSONSchema(zero)
	if err != nil {
		return nil, fmt.Errorf("failed to generate JSON schema: %w", err)
	}

	// First, check if the text contains extractable information
	validationPrompt := gollm.NewPrompt(fmt.Sprintf("分析以下文本是否包含足够的信息来提取结构化数据:\n\n%s\n\n如果文本包含可提取的信息，请回答'是'，如果不是，请回答'否'", text))
	validationPrompt.Apply(
		gollm.WithDirectives(
			"仅回答'是'或'否'",
			"如果文本包含足够的信息来填充大多数必填字段，请回答'是'",
			"如果文本不相关或缺少必要信息，请回答'否'",
		),
		gollm.WithOutput("单字回答：'是'或'否'"),
	)
	validationResponse, err := l.Generate(ctx, validationPrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to validate text content: %w", err)
	}
	if strings.TrimSpace(strings.ToLower(validationResponse)) != "yes" {
		return nil, fmt.Errorf("text does not contain enough extractable information")
	}

	// Proceed with extraction
	promptText := fmt.Sprintf("从给定的文本中提取以下信息:\n\n%s\n\nn请使用与此模式匹配的 JSON 对象进行响应:\n%s", text, string(schema))
	prompt := gollm.NewPrompt(promptText)
	prompt.Apply(append(opts,
		gollm.WithDirectives(
			"从文本中提取所有相关信息",
			"确保输出与提供的 JSON 模式完全匹配",
			"如果无法自信地填充某个字段，请将其保留为 null 或适当的空字符串/数组",
		),
		gollm.WithOutput("与提供的模式匹配的 JSON 对象"),
	)...)
	response, err := l.Generate(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate structured data: %w", err)
	}
	var result T
	if err := json.Unmarshal([]byte(response), &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	if err := gollm.Validate(&result); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}
	return &result, nil
}
