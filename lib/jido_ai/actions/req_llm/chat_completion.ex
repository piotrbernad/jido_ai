defmodule Jido.AI.Actions.ReqLlm.ChatCompletion do
  @moduledoc """
  Chat completion action using ReqLLM for multi-provider support.

  This action provides direct access to chat completion functionality across
  57+ providers through ReqLLM, replacing the LangChain-based implementation
  with lighter dependencies and broader provider support.

  ## Features

  - Multi-provider support (57+ providers via ReqLLM)
  - Tool/function calling capabilities
  - Response quality control with retry mechanisms
  - Support for various LLM parameters (temperature, top_p, etc.)
  - Structured error handling and logging
  - Streaming support (when provider allows)

  ## Usage

  ```elixir
  # Basic usage
  {:ok, result} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: %Jido.AI.Model{provider: :anthropic, model: "claude-3-sonnet-20240229"},
    prompt: Jido.AI.Prompt.new(:user, "What's the weather in Tokyo?")
  })

  # With function calling / tools
  {:ok, result} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: %Jido.AI.Model{provider: :openai, model: "gpt-4o"},
    prompt: prompt,
    tools: [Jido.Actions.Weather.GetWeather, Jido.Actions.Search.WebSearch],
    temperature: 0.2
  })

  # Streaming responses
  {:ok, stream} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: model,
    prompt: prompt,
    stream: true
  })

  Enum.each(stream, fn chunk ->
    IO.puts(chunk.content)
  end)
  ```

  ## Support Matrix

  Supports all providers available in ReqLLM (57+), including:
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Mistral, Cohere, Groq, and many more

  See ReqLLM documentation for full provider list.
  """
  use Jido.Action,
    name: "reqllm_chat_completion",
    description: "Chat completion action using ReqLLM",
    schema: [
      model: [
        type: {:custom, Jido.AI.Model, :validate_model_opts, []},
        required: true,
        doc:
          "The AI model to use (e.g., {:anthropic, [model: \"claude-3-sonnet-20240229\"]} or %Jido.AI.Model{})"
      ],
      prompt: [
        type: {:custom, Jido.AI.Prompt, :validate_prompt_opts, []},
        required: true,
        doc: "The prompt to use for the response"
      ],
      tools: [
        type: {:list, :atom},
        required: false,
        doc: "List of Jido.Action modules for function calling"
      ],
      max_retries: [
        type: :integer,
        default: 0,
        doc: "Number of retries for validation failures"
      ],
      temperature: [type: :float, default: 0.7, doc: "Temperature for response randomness"],
      max_tokens: [type: :integer, default: 1000, doc: "Maximum tokens in response"],
      top_p: [type: :float, doc: "Top p sampling parameter"],
      stop: [type: {:list, :string}, doc: "Stop sequences"],
      timeout: [type: :integer, default: 60_000, doc: "Request timeout in milliseconds"],
      stream: [type: :boolean, default: false, doc: "Enable streaming responses"],
      frequency_penalty: [type: :float, doc: "Frequency penalty parameter"],
      presence_penalty: [type: :float, doc: "Presence penalty parameter"],
      json_mode: [
        type: :boolean,
        default: false,
        doc: "Forces model to output valid JSON (provider-dependent)"
      ],
      verbose: [
        type: :boolean,
        default: false,
        doc: "Enable verbose logging"
      ]
    ]

  require Logger
  alias Jido.AI.Model
  alias Jido.AI.Prompt
  alias ReqLLM.Response, as: ReqResponse

  @impl true
  def on_before_validate_params(params) do
    with {:ok, model} <- validate_model(params.model),
         {:ok, prompt} <- Prompt.validate_prompt_opts(params.prompt) do
      {:ok, %{params | model: model, prompt: prompt}}
    else
      {:error, reason} ->
        Logger.error("ChatCompletion validation failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @impl true
  def run(params, _context) do
    # Validate required parameters exist
    with :ok <- validate_required_param(params, :model, "model"),
         :ok <- validate_required_param(params, :prompt, "prompt") do
      run_with_validated_params(params)
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp run_with_validated_params(params) do
    # Extract options from prompt if available
    prompt_opts =
      case params[:prompt] do
        %Prompt{options: options} when is_list(options) and length(options) > 0 ->
          Map.new(options)

        _ ->
          %{}
      end

    # Keep required parameters
    required_params = Map.take(params, [:model, :prompt, :tools])

    # Create a map with all optional parameters set to defaults
    # Priority: explicit params > prompt options > defaults
    params_with_defaults =
      %{
        temperature: 0.7,
        max_tokens: 1000,
        top_p: nil,
        stop: nil,
        timeout: 60_000,
        stream: false,
        max_retries: 0,
        frequency_penalty: nil,
        presence_penalty: nil,
        json_mode: false,
        verbose: false
      }
      # Apply prompt options over defaults
      |> Map.merge(prompt_opts)
      # Apply explicit params over prompt options
      |> Map.merge(
        Map.take(params, [
          :temperature,
          :max_tokens,
          :top_p,
          :stop,
          :timeout,
          :stream,
          :max_retries,
          :frequency_penalty,
          :presence_penalty,
          :json_mode,
          :verbose
        ])
      )
      # Always keep required params
      |> Map.merge(required_params)

    if params_with_defaults.verbose do
      Logger.info(
        "Running ReqLLM chat completion with params: #{inspect(params_with_defaults, pretty: true)}"
      )
    end

    with {:ok, model} <- validate_model(params_with_defaults.model),
         {:ok, messages} <- convert_messages(params_with_defaults.prompt),
         {:ok, req_options} <- build_req_llm_options(model, params_with_defaults),
         result <- call_reqllm(model, messages, req_options, params_with_defaults) do
      result
    else
      {:error, reason} ->
        Logger.error("Chat completion failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Private functions

  defp validate_required_param(params, key, name) do
    if Map.has_key?(params, key) do
      :ok
    else
      {:error, "Missing required parameter: #{name}"}
    end
  end

  defp validate_model(%ReqLLM.Model{} = model), do: {:ok, model}
  defp validate_model(%Model{} = model), do: Model.from(model)
  defp validate_model(spec) when is_tuple(spec), do: Model.from(spec)

  defp validate_model(other) do
    Logger.error("Invalid model specification: #{inspect(other)}")
    {:error, "Invalid model specification: #{inspect(other)}"}
  end

  defp convert_messages(prompt) do
    messages =
      Prompt.render(prompt)
      |> Enum.map(fn msg ->
        %{role: msg.role, content: msg.content}
      end)

    {:ok, messages}
  end

  defp build_req_llm_options(_model, params) do
    # Build base options
    base_opts =
      []
      |> add_opt_if_present(:temperature, params.temperature)
      |> add_opt_if_present(:max_tokens, params.max_tokens)
      |> add_opt_if_present(:top_p, params.top_p)
      |> add_opt_if_present(:stop, params.stop)
      |> add_opt_if_present(:frequency_penalty, params.frequency_penalty)
      |> add_opt_if_present(:presence_penalty, params.presence_penalty)

    # Add tools if provided
    opts_with_tools =
      case params[:tools] do
        tools when is_list(tools) and length(tools) > 0 ->
          # Convert tools to ReqLLM.Tool structs
          reqllm_tools =
            tools
            |> Enum.map(fn
              # Jido.Action module - convert to ReqLLM.Tool
              tool when is_atom(tool) ->
                convert_action_to_reqllm_tool(tool)

              # Already a ReqLLM.Tool struct
              %ReqLLM.Tool{} = tool ->
                tool

              # Map with function key (OpenAI format) - convert to ReqLLM.Tool
              %{function: %{name: name, description: description, parameters: parameters}} = tool
              when is_binary(name) and is_binary(description) ->
                convert_map_to_reqllm_tool(name, description, parameters)

              # Map with name, description, parameters (direct format)
              %{name: name, description: description, parameters: parameters} = tool
              when is_binary(name) and is_binary(description) and is_map(parameters) ->
                convert_map_to_reqllm_tool(name, description, parameters)

              # Any other map format - try to extract and convert
              tool when is_map(tool) ->
                case extract_tool_info_from_map(tool) do
                  {name, description, parameters} when is_binary(name) and is_binary(description) ->
                    convert_map_to_reqllm_tool(name, description, parameters)

                  _ ->
                    Logger.warning(
                      "Unable to convert tool map to ReqLLM.Tool: #{inspect(tool, limit: 3)}"
                    )
                    nil
                end

              other ->
                Logger.warning(
                  "Invalid tool format, expected Jido.Action module, ReqLLM.Tool struct, or map: #{inspect(other, limit: 3)}"
                )
                nil
            end)
            |> Enum.reject(&is_nil/1)
            |> Enum.filter(fn tool ->
              case tool do
                %ReqLLM.Tool{} -> true
                _ ->
                  Logger.error(
                    "Tool conversion failed - expected ReqLLM.Tool struct but got: #{inspect(tool, limit: 3)}"
                  )
                  false
              end
            end)

          Keyword.put(base_opts, :tools, reqllm_tools)

        _ ->
          base_opts
      end

    # ReqLLM handles authentication internally via environment variables
    {:ok, opts_with_tools}
  end

  defp convert_action_to_reqllm_tool(action_module) when is_atom(action_module) do
    name = get_action_name(action_module)
    description = get_action_description(action_module)
    parameter_schema = get_action_schema(action_module)

    # Create callback that executes the action
    callback = fn input ->
      # Convert input map to params format expected by Jido.Action
      params = normalize_input_to_params(input)
      context = %{}

      case action_module.run(params, context) do
        {:ok, result} -> {:ok, result}
        {:error, reason} -> {:error, reason}
        other -> {:ok, other}
      end
    end

    case ReqLLM.Tool.new(
           name: name,
           description: description,
           parameter_schema: parameter_schema,
           callback: callback
         ) do
      {:ok, tool} -> tool
      {:error, reason} ->
        Logger.error("Failed to create ReqLLM.Tool from #{inspect(action_module)}: #{inspect(reason)}")
        nil
    end
  end

  defp convert_map_to_reqllm_tool(name, description, parameters) do
    # Convert JSON Schema parameters to NimbleOptions format if needed
    parameter_schema =
      if is_map(parameters) do
        # Assume it's JSON Schema format, convert to NimbleOptions
        json_schema_to_nimble_options(parameters)
      else
        parameters
      end

    # Create a no-op callback since we don't have the original action
    callback = fn _input -> {:ok, %{message: "Tool executed"}} end

    case ReqLLM.Tool.new(
           name: name,
           description: description,
           parameter_schema: parameter_schema,
           callback: callback
         ) do
      {:ok, tool} -> tool
      {:error, reason} ->
        Logger.error("Failed to create ReqLLM.Tool from map: #{inspect(reason)}")
        nil
    end
  end

  defp extract_tool_info_from_map(%{name: name, description: description, parameters: parameters})
       when is_binary(name) and is_binary(description) do
    {name, description, parameters}
  end

  defp extract_tool_info_from_map(%{function: %{name: name, description: description, parameters: parameters}})
       when is_binary(name) and is_binary(description) do
    {name, description, parameters}
  end

  defp extract_tool_info_from_map(_), do: nil

  defp get_action_name(action_module) do
    if function_exported?(action_module, :name, 0) do
      action_module.name()
    else
      action_module
      |> Module.split()
      |> List.last()
      |> Macro.underscore()
    end
  end

  defp get_action_description(action_module) do
    if function_exported?(action_module, :description, 0) do
      action_module.description()
    else
      "No description available"
    end
  end

  defp get_action_schema(action_module) do
    if function_exported?(action_module, :schema, 0) do
      action_module.schema()
    else
      []
    end
  end

  defp normalize_input_to_params(input) when is_map(input) do
    # Convert atom keys to strings if needed, or keep as is
    Map.new(input, fn
      {key, value} when is_atom(key) -> {Atom.to_string(key), value}
      {key, value} -> {key, value}
    end)
  end

  defp normalize_input_to_params(input), do: input

  defp json_schema_to_nimble_options(%{"type" => "object", "properties" => properties} = schema) do
    required = Map.get(schema, "required", [])

    Enum.map(properties, fn {key, prop} ->
      atom_key = String.to_atom(key)

      opts = [
        type: json_type_to_nimble_type(Map.get(prop, "type", "string")),
        required: key in required
      ]

      opts =
        if Map.has_key?(prop, "description") do
          Keyword.put(opts, :doc, Map.get(prop, "description"))
        else
          opts
        end

      opts =
        if Map.has_key?(prop, "default") do
          Keyword.put(opts, :default, Map.get(prop, "default"))
        else
          opts
        end

      {atom_key, opts}
    end)
  end

  defp json_schema_to_nimble_options(_), do: []

  defp json_type_to_nimble_type("string"), do: :string
  defp json_type_to_nimble_type("integer"), do: :integer
  defp json_type_to_nimble_type("number"), do: :float
  defp json_type_to_nimble_type("boolean"), do: :boolean
  defp json_type_to_nimble_type("array"), do: {:list, :any}
  defp json_type_to_nimble_type("object"), do: :map
  defp json_type_to_nimble_type(_), do: :string

  defp add_opt_if_present(opts, _key, nil), do: opts
  defp add_opt_if_present(opts, key, value), do: Keyword.put(opts, key, value)

  defp call_reqllm(model, messages, req_options, params) do
    # Build model spec string from ReqLLM.Model
    model_spec = "#{model.provider}:#{model.model}"

    if params.stream do
      call_streaming(model_spec, messages, req_options)
    else
      call_standard(model_spec, messages, req_options)
    end
  end

  defp call_standard(model_id, messages, req_options) do
    case ReqLLM.generate_text(model_id, messages, req_options) do
      {:ok, response} ->
        # Use ReqLLM response directly
        format_response(response)

      {:error, error} ->
        {:error, error}
    end
  end

  defp call_streaming(model_id, messages, req_options) do
    opts_with_stream = Keyword.put(req_options, :stream, true)

    case ReqLLM.stream_text(model_id, messages, opts_with_stream) do
      {:ok, stream} ->
        # Return the stream wrapped in :ok tuple
        {:ok, stream}

      {:error, error} ->
        {:error, error}
    end
  end

  defp format_response(%ReqLLM.Response{} = response) do
    content = ReqLLM.Response.text(response) || ""
    tool_calls = ReqLLM.Response.tool_calls(response) || []

    formatted_tools =
      Enum.map(tool_calls, fn tool ->
        %{
          name: tool[:name] || tool["name"],
          arguments: tool[:arguments] || tool["arguments"],
          result: nil
        }
      end)

    {:ok, %{content: content, tool_results: formatted_tools}}
  end

  defp format_response(response) when is_map(response) and not is_struct(response) do
    content = response[:content] || response["content"] || ""
    tool_calls = response[:tool_calls] || response["tool_calls"] || []

    formatted_tools =
      if is_list(tool_calls) do
        Enum.map(tool_calls, fn tool ->
          %{
            name: tool[:name] || tool["name"],
            arguments: tool[:arguments] || tool["arguments"],
            result: nil
          }
        end)
      else
        []
      end

    {:ok, %{content: content, tool_results: formatted_tools}}
  end
end
