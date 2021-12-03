using SpinGlassExhaustive


function main(args)
    s = ArgParseSettings("description")
    @add_arg_table s begin
        "--steps", "-s"
          help = "number of samples per dimension"
          default = 10
          arg_type = Int
        "--dims", "-d"
          help = "dimensions"
          nargs = '*'
          default = [4, 16, 64]
          arg_type = Int
      end
    parsed_args = parse_args(s)
    steps = parsed_args["steps"]
    dims = parsed_args["dims"]
    savect(steps, dims)
  end
  
  main(ARGS)
  