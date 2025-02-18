// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/yockii/gollm_cn/llm"
)

// generateImprovedPrompt creates an enhanced version of a prompt based on its assessment
// and optimization history. It employs a dual-strategy approach, generating both
// incremental improvements and bold redesigns.
//
// The improvement process:
// 1. Analyzes previous assessment and optimization history
// 2. Generates two alternative improvements:
//   - Incremental: Refines existing approach
//   - Bold: Reimagines prompt structure
//
// 3. Evaluates expected impact of each version
// 4. Selects the version with higher potential impact
//
// The function considers:
// - Identified strengths and weaknesses
// - Historical optimization attempts
// - Task description and goals
// - Efficiency and clarity
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - prevEntry: Previous optimization entry containing prompt and assessment
//
// Returns:
//   - Improved prompt object
//   - Error if improvement generation fails
//
// Example improvement structure:
//
//	{
//	    "incrementalImprovement": {
//	        "input": "Refined prompt text...",
//	        "directives": ["Be more specific", "Add examples"],
//	        "examples": ["Example usage 1", "Example usage 2"],
//	        "reasoning": "Changes address clarity issues while maintaining strengths"
//	    },
//	    "boldRedesign": {
//	        "input": "Completely restructured prompt...",
//	        "directives": ["New approach", "Different perspective"],
//	        "examples": ["New example 1", "New example 2"],
//	        "reasoning": "Novel approach potentially offers better results"
//	    },
//	    "expectedImpact": {
//	        "incremental": 16.5,
//	        "bold": 18.0
//	    }
//	}
func (po *PromptOptimizer) generateImprovedPrompt(ctx context.Context, prevEntry OptimizationEntry) (*llm.Prompt, error) {
	recentHistory := po.recentHistory()
	improvePrompt := llm.NewPrompt(fmt.Sprintf(`
		基于以下评估和最近的历史记录，生成整个提示词结构的改进版本：

		先前的提示词: %+v
		评估: %+v

		最近的历史记录:
		%+v

		任务描述: %s
		优化目标: %s

		在生成改进时，请考虑最近的历史记录。

		提供两个改进的提示词版本：
		1. 渐进式改进
		2. 大胆的重新设计

		重要提示：仅以原始 JSON 对象回复。请勿使用任何 Markdown 格式、代码块或反引号。
		JSON 对象应具有以下结构：
		{
			"incrementalImprovement": {
				"input": "改进的提示词文本",
				"directives": ["指令1", "指令2", ...],
				"examples": ["示例1", "示例2", ...],
				"reasoning": "变更的解释及其与评估的联系"
			},
			"boldRedesign": {
				"input": "重新设计的提示词文本",
				"directives": ["指令1", "指令2", ...],
				"examples": ["示例1", "示例2", ...],
				"reasoning": "对新方法的解释及其潜在优势"
			},
			"expectedImpact": {
				"incremental": number,
				"bold": number
			}
		}

		对于每个改进：
		- 直接解决评估中发现的弱点。
		- 以已识别的优势为基础。
		- 确保与任务描述和优化目标保持一致。
		- 力求语言使用的效率。
		- 使用清晰、无术语的语言。
		- 为主要变更提供简要的理由。
		- 以 0 到 20 的等级对每个版本的预期影响进行评级。

		在提交之前，请仔细检查您的回复是否为有效的 JSON。
	`, prevEntry.Prompt, prevEntry.Assessment, recentHistory, po.taskDesc, po.optimizationGoal))

	// Log the improvement request for debugging
	po.debugManager.LogPrompt(improvePrompt.String())

	// Generate improvements using LLM
	response, err := po.llm.Generate(ctx, improvePrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate improved prompt: %w", err)
	}

	// Log the raw response for debugging
	po.debugManager.LogResponse(response)

	// Extract and parse JSON response
	cleanedResponse := cleanJSONResponse(response)

	var improvedPrompts struct {
		IncrementalImprovement llm.Prompt `json:"incrementalImprovement"`
		BoldRedesign           llm.Prompt `json:"boldRedesign"`
		ExpectedImpact         struct {
			Incremental float64 `json:"incremental"`
			Bold        float64 `json:"bold"`
		} `json:"expectedImpact"`
	}

	err = json.Unmarshal([]byte(cleanedResponse), &improvedPrompts)
	if err != nil {
		return nil, fmt.Errorf("failed to parse improved prompts: %w", err)
	}

	// Select the improvement with higher expected impact
	if improvedPrompts.ExpectedImpact.Bold > improvedPrompts.ExpectedImpact.Incremental {
		return &improvedPrompts.BoldRedesign, nil
	}
	return &improvedPrompts.IncrementalImprovement, nil
}
